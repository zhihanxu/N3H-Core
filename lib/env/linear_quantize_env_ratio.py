# Code for "[HAQ: Hardware-Aware Automated Quantization with Mixed Precision"
# Kuan Wang*, Zhijian Liu*, Yujun Lin*, Ji Lin, Song Han
# {kuanwang, zhijian, yujunlin, jilin, songhan}@mit.edu

import os
import time
import math
import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy
import torch.optim as optim
from progress.bar import Bar

from lib.utils.utils import AverageMeter, accuracy, prGreen, measure_model
from lib.utils.data_utils import get_dataset
from lib.utils.quantize_utils_ratio import QConv2d, QLinear, calibrate

class LinearQuantizeEnv:
    def __init__(self, model, pretrained_model, data, data_root, compress_ratio, args, n_data_worker=16,
                 batch_size=256, float_bit=8, is_model_pruned=False):
        # default setting nn.Conv2d, nn.Linear
        self.quantizable_layer_types = [QConv2d, QLinear]
        
        # save options
        self.model = model
        self.model_for_measure = deepcopy(model)
        self.model_name = args.arch
        self.cur_ind = 0
        self.strategy = []  # quantization strategy

        self.finetune_lr = args.finetune_lr
        self.optimizer = optim.SGD(model.parameters(), lr=args.finetune_lr, momentum=0.9, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.pretrained_model = pretrained_model
        self.n_data_worker = n_data_worker
        self.batch_size = batch_size
        self.data_type = data   #cifar10
        self.data_root = data_root
        self.compress_ratio = compress_ratio
        self.is_model_pruned = is_model_pruned
        self.val_size = args.val_size
        self.train_size = args.train_size
        self.finetune_gamma = args.finetune_gamma
        self.finetune_lr = args.finetune_lr
        self.finetune_flag = args.finetune_flag
        self.finetune_epoch = args.finetune_epoch   #default 1

        # options from args
        self.min_bit = args.min_bit
        self.max_bit = args.max_bit
        self.float_bit = float_bit * 1.
        self.last_weight_action = self.max_bit
        self.last_activation_action = self.max_bit
        self.last_ratio_action = 0
        self.action_radio_button = True

        self.is_inception = args.arch.startswith('inception')
        self.is_imagenet = ('ILSVRC2012' or 'imagenet100' in data)
        
        self.use_top5 = args.use_top5

        # init reward
        self.best_reward = -math.inf

        # prepare data
        self._init_data()

        # build indexs
        self._build_index() 
        self.n_quantizable_layer = len(self.quantizable_idx)

        self.model.load_state_dict(self.pretrained_model, strict=True)

        # self.org_acc = self._finetune(self.train_loader, self.model, epochs=self.finetune_epoch, verbose=False) 
        self.org_acc = self._validate(self.val_loader, self.model)
        # self.org_acc = 99
        # self.ema_acc = 0
        
        # build embedding (static part), same as pruning
        self._build_state_embedding()
        
        self.lut = get_costlookuptable()
        self.ratio_lut = get_ratiolookuptable()
        
        # self.org_cost = self.cal_bismo_res18(i=-1, ratio=0.28, Ab=8, Bb=8, max_flag=True)
        self.org_cost = cal_dsp_res18(i=-1, ratio=1, max_flag=True)
        # self.min_cost = cal_dsp(i=-1, ratio=0.43, max_flag=True)
        self.min_cost = round(self.cal_bismo_res18(i=-1, ratio=0.42, Ab=2, Bb=2, max_flag=True),2)

        # sanity check
        # assert self.compress_ratio > self.min_cost / self.org_cost, \
        #     'Error! You can make achieve compress_ratio smaller than min_bit!'

        # restore weight
        self.reset()
        print('=> original acc: {:.3f}% on split dataset(train: %7d, val: %7d )'.format(self.org_acc, self.train_size, self.val_size))
        print('=> original cost: {:.4f}', self.org_cost)

    def adjust_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.finetune_gamma
    
    def step(self, action):

        action = self._action_wall(action)

        if self.action_radio_button: 
            self.last_weight_action = action    
        else:
            self.last_activation_action = action
            self.last_ratio_action = self.ratio_lut[self.last_activation_action-2][self.last_weight_action-2] 
            self.strategy.append([self.last_weight_action, self.last_activation_action, self.last_ratio_action])  # save action to strategy

        # all the actions are made
        if self._is_final_layer() and (not self.action_radio_button): #ratio:-1    1 0 -1
            target = self.compress_ratio * self.org_cost #compress相当于compress了cost
            target = round(target,2)
            self._keep_first_last_layer()
            # self._keep_shorcut()
            
            cost = self._cur_cost()
            
            print('\nstrategy: ', self.strategy, 'min_cost:', self.min_cost, 'target:', target, 'current_cost:', cost)
            
            assert len(self.strategy) == len(self.quantizable_idx)
            cost_ratio = cost / self.org_cost
            
            self._set_mixed_precision(quantizable_idx=self.quantizable_idx, strategy=self.strategy)
            
            if cost > target:
                reward = self.reward(0, target, cost)
                acc = 0
            else:
                self.model = calibrate(self.model, self.train_loader)       # train_loader
                if not self.finetune_flag:                                  # default true
                    acc = self._finetune(self.train_loader, self.model, epochs=self.finetune_epoch, verbose=False)  
                    # train_acc = self._finetune(self.train_loader, self.model, epochs=self.finetune_epoch, verbose=False)
                    # acc = self._validate(self.val_loader, self.model)
                else:
                    acc = self._validate(self.train_loader, self.model)   #validate
                    # self.ema_acc = EMA(self.ema_acc, 1/100, acc)
                    # print(self.ema_acc)
                reward = self.reward(acc, target, cost)
            info_set = {'cost_ratio': cost_ratio, 'accuracy': acc, 'cost': cost}

            if reward > self.best_reward:
                self.best_reward = reward
                prGreen('New best policy: {}, reward: {:.3f}, acc: {:.3f}, cost_ratio: {:.3f}'.format(
                    self.strategy, self.best_reward, acc, cost_ratio))

            obs = self.layer_embedding[self.cur_ind, :].copy()  # actually the same as the last state
            done = True
            self.action_radio_button = not self.action_radio_button
            return obs, reward, done, info_set

        cost = self._cur_cost()
        info_set = {'cost': cost}
        reward = 0
        done = False   

        if self.action_radio_button:
            self.layer_embedding[self.cur_ind][-1] = 0.0   #build index of next layer
        else:
            self.cur_ind += 1
            self.layer_embedding[self.cur_ind][-1] = 1.0
        self.layer_embedding[self.cur_ind][-2] = float(action) / float(self.max_bit)
        self.layer_embedding[self.cur_ind][-1] = float(self.action_radio_button)    #1 0 -1
        # build next state (in-place modify)
        obs = self.layer_embedding[self.cur_ind, :].copy()
        self.action_radio_button = self.action_radio_button - 1 if self.action_radio_button > -1 else 1
        return obs, reward, done, info_set
        
    def reward(self, acc, target, cost):
        if cost > target:
            return -cost/target
        else:
            return (acc-self.org_acc)*0.01  # self.ema_acc    (acc-self.org_acc + cost/target)*0.01   (acc-self.org_acc)*0.1

    def reset(self):
        # restore env by loading the pretrained model
        self.model.load_state_dict(self.pretrained_model, strict=False)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.finetune_lr, momentum=0.9, weight_decay=4e-5)
        self.cur_ind = 0
        self.strategy = []  # quantization strategy
        obs = self.layer_embedding[0].copy()
        return obs

    def _is_final_layer(self):
        return self.cur_ind == len(self.quantizable_idx) - 1

    def _keep_first_last_layer(self):
        self.strategy[0][0] = 8
        # input image is already 8 bit -1
        self.strategy[0][1] = 8
        self.strategy[0][2] = 0.73
        self.strategy[-1][0] = 8
        self.strategy[-1][1] = 8
        self.strategy[-1][2] = 0.73
        
    def _keep_shorcut(self):
        self.strategy[9][0] = 8
        # input image is already 8 bit -1
        self.strategy[9][1] = 8
        self.strategy[9][2] = 0.51
        self.strategy[16][0] = 8
        self.strategy[16][1] = 8
        self.strategy[16][2] = 0.51

    #action space
    def _action_wall(self, action):
        assert len(self.strategy) == self.cur_ind
        # limit the action to certain range
        action = float(action)
        min_bit, max_bit = self.bound_list[self.cur_ind]
        lbound, rbound = min_bit - 0.5, max_bit + 0.5  # same stride length for each bit
        action = (rbound - lbound) * action + lbound
        action = int(np.round(action, 0))
        return action

    def _set_mixed_precision(self, quantizable_idx, strategy):
        assert len(quantizable_idx) == len(strategy), \
            'You should provide the same number of bit setting as layer list for weight quantization!'
        quantize_layer_bit_dict = {n: b for n, b in zip(quantizable_idx, strategy)}

        for i, layer in enumerate(self.model.modules()):
            if i not in quantizable_idx:
                continue
            else:
                layer.w_bit = quantize_layer_bit_dict[i][0]
                layer.a_bit = quantize_layer_bit_dict[i][1]
                layer.ratio = quantize_layer_bit_dict[i][2]

    #current cost
    def _cur_cost(self):
        cur_cost = 0
        for i, n_bit in enumerate(self.strategy):
            cur_cost += self.cal_lat_res18(i, n_bit[-1], n_bit[1], n_bit[0])
        cur_cost = round(cur_cost, 2)
        return cur_cost

    def _init_data(self):
        # self.train_loader, self.val_loader, n_class = get_split_train_dataset(
        #     self.data_type, self.batch_size, self.n_data_worker, data_root=self.data_root,
        #     val_size=self.val_size, train_size=self.train_size, for_inception=self.is_inception)
        self.train_loader, self.val_loader, n_class = get_dataset(
            self.data_type, self.batch_size, self.n_data_worker, data_root=self.data_root,
            for_inception=self.is_inception)

    def _build_index(self):
        self.quantizable_idx = []
        self.bound_list = []
        for i, m in enumerate(self.model.modules()):
            if type(m) in self.quantizable_layer_types:
                self.quantizable_idx.append(i)
                self.bound_list.append((self.min_bit, self.max_bit))
        print('=> Final bound list: {}'.format(self.bound_list))    

    def _build_state_embedding(self):
        # measure model for cifar 32x32 input
        if self.is_imagenet:
            measure_model(self.model_for_measure, 224, 224)
        else:
            measure_model(self.model_for_measure, 32, 32)
        # build the static part of the state embedding
        layer_embedding = []
        module_list = list(self.model_for_measure.modules())
        for i, ind in enumerate(self.quantizable_idx):
            m = module_list[ind]
            this_state = []
            if type(m) == nn.Conv2d or type(m) == QConv2d:  #conv layer
                this_state.append([int(m.in_channels == m.groups)])  # layer type, 1 for conv_dw
                this_state.append([m.in_channels])  # in channels
                this_state.append([m.out_channels])  # out channels
                this_state.append([m.stride[0]])  # stride
                this_state.append([m.kernel_size[0]])  # kernel size
                this_state.append([np.prod(m.weight.size())])  # weight size
                this_state.append([m.in_w*m.in_h])  # input feature_map_size
            elif type(m) == nn.Linear or type(m) == QLinear:    #activation layer
                this_state.append([0.])  # layer type, 0 for fc
                this_state.append([m.in_features])  # in channels
                this_state.append([m.out_features])  # out channels
                this_state.append([0.])  # stride
                this_state.append([1.])  # kernel size
                this_state.append([np.prod(m.weight.size())])  # weight size
                this_state.append([m.in_w*m.in_h])  # input feature_map_size

            this_state.append([i])  # index
            this_state.append([1.])  # bits, 1 is the max bit
            this_state.append([1.])  # action radio button, 1 is the weight action, -1 is the ratio
            layer_embedding.append(np.hstack(this_state))

        # normalize the state to [0,1]
        layer_embedding = np.array(layer_embedding, 'float')
        print('=> shape of embedding (n_layer * n_dim): {}'.format(layer_embedding.shape))
        assert len(layer_embedding.shape) == 2, layer_embedding.shape
        for i in range(layer_embedding.shape[1]):
            fmin = min(layer_embedding[:, i])
            fmax = max(layer_embedding[:, i])
            if fmax - fmin > 0:
                layer_embedding[:, i] = (layer_embedding[:, i] - fmin) / (fmax - fmin)

        self.layer_embedding = layer_embedding


    def _finetune(self, train_loader, model, epochs=1, verbose=True):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        best_acc = 0.

        # switch to train mode
        model.train()
        end = time.time()
        t1 = time.time()
        bar = Bar('train:', max=len(train_loader))
        for epoch in range(epochs):
            for i, (inputs, targets) in enumerate(train_loader):
                # if i == 30:
                #     break
                input_var, target_var = inputs.cuda(), targets.cuda()

                # measure data loading time
                data_time.update(time.time() - end)

                # compute output
                output = model(input_var)
                loss = self.criterion(output, target_var)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                # compute gradient
                self.optimizer.zero_grad()
                loss.backward()

                # do SGD step
                self.optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                if i % 1 == 0:
                    bar.suffix = \
                        '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                        'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                            batch=i + 1,
                            size=len(train_loader),
                            data=data_time.val,
                            bt=batch_time.val,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss=losses.avg,
                            top1=top1.avg,
                            top5=top5.avg,
                        )
                    bar.next()
            bar.finish()

            if self.use_top5:
                if top5.avg > best_acc:
                    best_acc = top5.avg
            else:
                if top1.avg > best_acc:
                    best_acc = top1.avg
            self.adjust_learning_rate()
        t2 = time.time()
        if verbose:
            print('* Test loss: %.3f  top1: %.3f  top5: %.3f  time: %.3f' % (losses.avg, top1.avg, top5.avg, t2-t1))
        return best_acc

    #validate
    def _validate(self, val_loader, model, verbose=False):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        t1 = time.time()
        with torch.no_grad():
            # switch to evaluate mode
            model.eval()

            end = time.time()
            bar = Bar('valid:', max=len(val_loader))
            for i, (inputs, targets) in enumerate(val_loader):
                # measure data loading time
                data_time.update(time.time() - end)
                # print(inputs.shape)

                input_var, target_var = inputs.cuda(), targets.cuda()

                # compute output
                output = model(input_var)
                loss = self.criterion(output, target_var)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                # plot progress
                if i % 1 == 0:
                    bar.suffix = \
                        '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                        'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                            batch=i + 1,
                            size=len(val_loader),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss=losses.avg,
                            top1=top1.avg,
                            top5=top5.avg,
                        )
                    bar.next()
            bar.finish()
        t2 = time.time()
        if verbose:
            print('* Test loss: %.3f  top1: %.3f  top5: %.3f  time: %.3f' % (losses.avg, top1.avg, top5.avg, t2-t1))
        if self.use_top5:
            return top5.avg
        else:
            return top1.avg
    
    def cal_bismo(self, i, ratio, Ab, Bb, max_flag=False):
        if max_flag == True:
            # max_lat = (4816896*(1-ratio)*(self.lut[Ab-2][Bb-2])+9437184*(1-ratio)*(self.lut[Ab-2][Bb-2])*16+2048000*(1-ratio)*(self.lut[Ab-2][Bb-2]))/131072
            max_lat = (1327104*(1-ratio)*(self.lut[Ab-2][Bb-2])+37748736*(1-ratio)*(self.lut[Ab-2][Bb-2])*7+18874368*(1-ratio)*(self.lut[Ab-2][Bb-2])+
                 75497472*(1-ratio)*(self.lut[Ab-2][Bb-2])*5+150994944*(1-ratio)*(self.lut[Ab-2][Bb-2])*5+40960*(1-ratio)*(self.lut[Ab-2][Bb-2])
                 +4194304*(1-ratio)*(self.lut[Ab-2][Bb-2])+8388608*(1-ratio)*(self.lut[Ab-2][Bb-2]))/131072
            return max_lat
        if i == 0:
            bismo_lat = 1327104*(1-ratio)*(self.lut[Ab-2][Bb-2])/131072
        elif i == 14 or (i > 0 and i <= 6):
            bismo_lat = 37748736*(1-ratio)*(self.lut[Ab-2][Bb-2])/131072
        elif i == 7:
            bismo_lat = 18874368*(1-ratio)*(self.lut[Ab-2][Bb-2])/131072
        elif i == 9:
            bismo_lat = 4194304*(1-ratio)*(self.lut[Ab-2][Bb-2])/131072       
        elif i == 8 or (i > 9 and i <= 13):
            bismo_lat = 75497472*(1-ratio)*(self.lut[Ab-2][Bb-2])/131072
        elif i == 15 or (i > 16 and i <= 20):
            bismo_lat = 150994944*(1-ratio)*(self.lut[Ab-2][Bb-2])/131072
        elif i == 16:
            bismo_lat = 8388608*(1-ratio)*(self.lut[Ab-2][Bb-2])/131072
        else:
            bismo_lat = 40960*(1-ratio)*(self.lut[Ab-2][Bb-2])/131072
        return bismo_lat

    def cal_bismo_res18(self, i, ratio, Ab, Bb, max_flag=False):
        if max_flag == True:
            max_lat = (1769472*(1-ratio)+37748736*(1-ratio)*13+18874368*(1-ratio)*3+2097152*(1-ratio)*3+5120*(1-ratio))*(self.lut[Ab-2][Bb-2])/8192
            return max_lat
        if i == 0:
            bismo_lat = 1769472*(1-ratio)*(self.lut[Ab-2][Bb-2])/8192
        elif i == 6 or (i > 0 and i <= 4) or (i > 7 and i <= 9) or i == 11 or (i > 12 and i <= 14) or i == 16 or (i > 17 and i <= 19):
            bismo_lat = 37748736*(1-ratio)*(self.lut[Ab-2][Bb-2])/8192
        elif i == 5 or i == 10 or i == 15:
            bismo_lat = 18874368*(1-ratio)*(self.lut[Ab-2][Bb-2])/8192
        elif i == 7 or i == 12 or i == 17:
            bismo_lat = 2097152*(1-ratio)*(self.lut[Ab-2][Bb-2])/8192      
        else:
            bismo_lat = 5120*(1-ratio)*(self.lut[Ab-2][Bb-2])/8192
        return bismo_lat
    
    def cal_lat(self, i, ratio, Ab, Bb):
        if cal_dsp(i, ratio) < self.cal_bismo(i, ratio, Ab, Bb):
            latency = self.cal_bismo(i, ratio, Ab, Bb)
        else:
            latency = cal_dsp(i, ratio,)
        return latency 
    
    def cal_lat_res18(self, i, ratio, Ab, Bb):
        if cal_dsp_res18(i, ratio) < self.cal_bismo_res18(i, ratio, Ab, Bb):
            latency = self.cal_bismo_res18(i, ratio, Ab, Bb)
        else:
            latency = cal_dsp_res18(i, ratio)
        return latency 
    
def get_ratiolookuptable():
    lookup_table_folder = 'lib/simulator/lookup_tables/'
    os.makedirs(lookup_table_folder, exist_ok=True)
    
    fname = lookup_table_folder + 'ratio2.npy'
    print('load latency table : ', fname)
    ratio_list = np.load(fname)
    return ratio_list.copy()
   
def get_costlookuptable():
    lookup_table_folder = 'lib/simulator/lookup_tables/'
    os.makedirs(lookup_table_folder, exist_ok=True)
    
    fname = lookup_table_folder + 'bismo_latency.npy'
    print('load latency table : ', fname)
    latency_list = np.load(fname)
    return latency_list.copy()

def cal_dsp(i, ratio, max_flag=False):
    if max_flag == True:
        # max_lat = (4816896*ratio*35850+9437184*ratio*35850*16+2048000*ratio*35850)/65536/32
        max_lat = (1327104*ratio*35850+37748736*ratio*35850*7+18874368*ratio*35850+
                 75497472*ratio*35850*5+150994944*ratio*35850*5+40960*ratio*35850
                 +4194304*ratio*35850+8388608*ratio*35850)/65536/32
        return max_lat
    if i == 0:
        dsp_lat = 1327104*ratio*35850/65536/32
    elif i == 14 or (i > 0 and i <= 6):
        dsp_lat = 37748736*ratio*35850/65536/32
    elif i == 7:
        dsp_lat = 18874368*ratio*35850/65536/32
    elif i == 9:
        dsp_lat = 4194304*ratio*35850/65536/32     
    elif i == 8 or (i > 9 and i <= 13):
        dsp_lat = 75497472*ratio*35850/65536/32
    elif i == 15 or (i > 16 and i <= 20):
        dsp_lat = 150994944*ratio*35850/65536/32
    elif i == 16:
        dsp_lat = 8388608*ratio*35850/65536/32
    else:
        dsp_lat = 40960*ratio*35850/65536/32
    return dsp_lat

def cal_dsp_res18(i, ratio, max_flag=False):
    if max_flag == True:
        max_lat = (1769472*ratio+37748736*ratio*13+18874368*ratio*3+2097152*ratio*3+5120*ratio)*4/256
        return max_lat
    if i == 0:
        dsp_lat = 1769472*ratio*4/256
    elif i == 6 or (i > 0 and i <= 4) or (i > 7 and i <= 9) or i == 11 or (i > 12 and i <= 14) or i == 16 or (i > 17 and i <= 19):
        dsp_lat = 37748736*ratio*4/256
    elif i == 5 or i == 10 or i == 15:
        dsp_lat = 18874368*ratio*4/256
    elif i == 7 or i == 12 or i == 17:
        dsp_lat = 2097152*ratio*4/256     
    else:
        dsp_lat = 5120*ratio*4/256
    return dsp_lat
