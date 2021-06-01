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
from lib.utils.quantize_utils_ru import QConv2d, QLinear, calibrate

class LinearQuantizeEnv:
    def __init__(self, model, pretrained_model, data, data_root, target_latency, args, n_data_worker=16,
                 batch_size=256, float_bit=8):
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
        self.target_latency = target_latency
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
        self.weight_action = self.max_bit
        self.activation_action = self.max_bit
        self.action_radio_button = True
        
        self.M = 0
        self.K = 0
        self.N = 0
        self.Bm = 0
        self.Lmem = 0
        self.Rmem = 0
        self.ru = []
        self.ratio_action = 0

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
        
        # build state embedding
        self._build_state_embedding()
        
        self.max_cost = 198.99        
        self.min_cost = 22.06             

        # sanity check
        assert self.target_latency > self.min_cost

        # restore weight
        self.reset()
        print('=> original acc: {:.3f}% on split dataset(train: %7d, val: %7d )'.format(self.org_acc, self.train_size, self.val_size))
        print('=> max cost: {:.4f}', self.max_cost)

    def adjust_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.finetune_gamma
    
    def step(self, action):
        if self.cur_ind == 0: 
            action = self._action_wall2(64, 1, 4, action)
            self.K = action
            self.ru.append(self.K)
            # self.layer_embedding[0][9] = self.K
        elif self.cur_ind == 1:
            action = self._action_wall2(1, 1, 50, action)
            self.M = action
            self.ru.append(self.M)
            # self.layer_embedding[0][10] = self.M
        elif self.cur_ind == 2:
            action = self._action_wall2(1, 1, 50, action)
            self.N = action
            self.ru.append(self.N)
            # self.layer_embedding[0][11] = self.N
        elif self.cur_ind == 3:
            action = self._action_wall2(1024, 1, 50, action)
            self.Bm = action
            self.ru.append(self.Bm)
            # self.layer_embedding[0][12] = self.Bm
        elif self.cur_ind == 4:
            action = self._action_wall2(1024, 1, 25, action)
            self.Lmem = action
            self.ru.append(self.Lmem)
            # self.layer_embedding[0][13] = self.Lmem
        elif self.cur_ind == 5:
            action = self._action_wall2(1024, 1, 4, action)
            self.Rmem = action
            self.ru.append(self.Rmem)
            # self.layer_embedding[0][14] = self.Rmem
        else:
            if self.action_radio_button:
                self.weight_action = self._action_wall(2, 8, action)
            else:
                self.activation_action = self._action_wall(2, 4, action)
                self.ratio_action = round(self.cal_ratio(self.cur_ind - 6, self.activation_action, self.weight_action),2)
                self.strategy.append([self.activation_action, self.weight_action, self.ratio_action])  # save action to strategy

        # all the actions are made
        if self._is_final_layer() and (not self.action_radio_button):
            target = self.target_latency
            target = round(target,2)
            self._keep_first_last_layer()

            cost = self._cur_cost()
            
            print('\nResource setting: ', self.ru)
            print('strategy: ', self.strategy, 'min_cost:', self.min_cost, 'target:', target, 'current_cost:', cost)
            assert len(self.strategy) == len(self.quantizable_idx)
            cost_ratio = cost / self.max_cost
            
            self._set_mixed_precision(quantizable_idx=self.quantizable_idx, strategy=self.strategy)
            
            if cost > target:
                reward = self.reward(0, target, cost)
                acc = 0
            else:
                self.model = calibrate(self.model, self.train_loader)       
                if not self.finetune_flag:                                  
                    acc = self._finetune(self.train_loader, self.model, epochs=self.finetune_epoch, verbose=False)  
                else:
                    acc = self._validate(self.train_loader, self.model)   
                reward = self.reward(acc, target, cost)
            info_set = {'cost_ratio': cost_ratio, 'accuracy': acc, 'cost': cost}

            if reward > self.best_reward:
                self.best_reward = reward
                prGreen('New best policy: {}, reward: {:.3f}, acc: {:.3f}, cost_ratio: {:.3f}'.format(
                    self.strategy, self.best_reward, acc, cost_ratio))

            obs = self.layer_embedding.copy()  # actually the same as the last state
            done = True
            self.cur_ind = 0
            self.action_radio_button = not self.action_radio_button
            return obs, reward, done, info_set
        
        if self.cur_ind < 6 or self.action_radio_button == 1:
            cost = 0
        elif self.cur_ind > 5 and self.action_radio_button == 0:
            cost = self._cur_cost()
        info_set = {'cost': cost}
        reward = 0
        done = False   

        if self.cur_ind < 6:
            self._build_state_embedding()
            self.layer_embedding[0][-1] = float(action)    
            # print(self.layer_embedding)
            obs = self.layer_embedding.copy()
            self.cur_ind += 1
        else:
            self._build_state_embedding()
            self.layer_embedding[0][-2] = self.ratio_action
            self.layer_embedding[0][-1] = float(action)
            # print(self.layer_embedding)
            obs = self.layer_embedding.copy()
            self.action_radio_button = not self.action_radio_button
            if self.action_radio_button:
                self.cur_ind += 1
        return obs, reward, done, info_set
        
    def reward(self, acc, target, cost):
        if cost > target:
            return -cost/target
        else:
            return (acc-self.org_acc)*0.01

    def reset(self):
        # restore env by loading the pretrained model
        self.model.load_state_dict(self.pretrained_model, strict=False)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.finetune_lr, momentum=0.9, weight_decay=4e-5)
        self.cur_ind = 0
        self.strategy = []  # quantization strategy
        self.ru = []
        obs = self.layer_embedding.copy()
        return obs

    def _is_final_layer(self):
        return self.cur_ind == len(self.quantizable_idx) - 1 + 6

    def _keep_first_last_layer(self):
        self.strategy[0][0] = 8
        # input image is already 8 bit -1
        self.strategy[0][1] = 8
        # self.strategy[0][2] = 0.03  #cifar
        self.strategy[0][2] = 0.25
        self.strategy[-1][0] = 8
        self.strategy[-1][1] = 8
        self.strategy[-1][2] = 1

    #action space
    def _action_wall(self, min_bit, max_bit, action):
        # limit the action to certain range
        action = float(action)
        lbound, rbound = min_bit - 0.5, max_bit + 0.5  # same stride length for each bit
        action = (rbound - lbound) * action + lbound
        action = int(np.round(action, 0))
        return action
    
    def _action_wall2(self, c, vmin, vmax, action):
        action = float(action)
        action = (vmax - vmin) * action + vmin
        action = c * int(np.round(action, 0))
        return action

    def _set_mixed_precision(self, quantizable_idx, strategy):
        assert len(quantizable_idx) == len(strategy), \
            'You should provide the same number of bit setting as layer list for weight quantization!'
        quantize_layer_bit_dict = {n: b for n, b in zip(quantizable_idx, strategy)}

        for i, layer in enumerate(self.model.modules()):
            if i not in quantizable_idx:
                continue
            else:
                layer.a_bit = quantize_layer_bit_dict[i][0]
                layer.w_bit = quantize_layer_bit_dict[i][1]
                layer.ratio = quantize_layer_bit_dict[i][2]

    #current cost
    def _cur_cost(self):
        cur_cost = 0
        for i, n_bit in enumerate(self.strategy):
            cur_cost += self.cal_lat(i, n_bit[2], n_bit[0], n_bit[1])
        cur_cost = round(cur_cost, 2)
        return cur_cost

    def _init_data(self):
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
        
        if self.cur_ind < 6:
            this_state = []
            this_state.append([0.] * 10)
            this_state.append([self.K])  # K
            this_state.append([self.M])  # M
            this_state.append([self.N])  # N
            this_state.append([self.Bm])  # Bm
            this_state.append([self.Lmem])  # Lmem
            this_state.append([self.Rmem])  # Rmem
            
            this_state.append([0.])  # ratio
            this_state.append([0.])  # action     
        else:
            i = self.cur_ind - 6
            ind = self.quantizable_idx[i]
            m = module_list[ind]
            this_state = []
            
            this_state.append([1.])     # function index
            this_state.append([i])      # layer index
            if type(m) == nn.Conv2d or type(m) == QConv2d:  #conv layer
                # this_state.append([int(m.in_channels == m.groups)])  # layer type, 1 for conv_dw
                this_state.append([m.in_channels])  # in channels
                this_state.append([m.out_channels])  # out channels
                this_state.append([m.stride[0]])  # stride
                this_state.append([m.kernel_size[0]])  # kernel size
                this_state.append([np.prod(m.weight.size())])  # weight size
                this_state.append([m.in_w*m.in_h])  # input feature_map_size
                this_state.append([int(self.cur_ind==8 or self.cur_ind==13 or self.cur_ind==18)]) #downsampling
                this_state.append([self.action_radio_button])
            elif type(m) == nn.Linear or type(m) == QLinear:    #activation layer
                this_state.append([m.in_features])  # in channels
                this_state.append([m.out_features])  # out channels
                this_state.append([0.])  # stride
                this_state.append([1.])  # kernel size
                this_state.append([np.prod(m.weight.size())])  # weight size
                this_state.append([m.in_w*m.in_h])  # input feature_map_size
                this_state.append([0.])
                this_state.append([self.action_radio_button])

            # resource utilization
            this_state.append([self.K])  # K
            this_state.append([self.M])  # M
            this_state.append([self.N])  # N
            this_state.append([self.Bm])  # Bm
            this_state.append([self.Lmem])  # Lmem
            this_state.append([self.Rmem])  # Rmem
            
            this_state.append([0.])  # ratio
            this_state.append([0.])  # action
                
        layer_embedding.append(np.hstack(this_state))

        # normalize the state to [0,1]
        layer_embedding = np.array(layer_embedding, 'float')
        # print('=> shape of embedding (n_layer * n_dim): {}'.format(layer_embedding.shape))
        # assert len(layer_embedding.shape) == 2, layer_embedding.shape
        for i in range(layer_embedding.shape[0]):
            fmin = min(layer_embedding[i])
            fmax = max(layer_embedding[i])
            if fmax - fmin > 0:
                layer_embedding[i] = (layer_embedding[i] - fmin) / (fmax - fmin)

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

    def cal_bismo(self, i, ratio, Ab, Bb):
        
        RMEM = 1024 * 1 # NO impact
        LMEM = self.Bm  # 1024 * 1

        M = self.M  # 8
        K = self.K  # 128
        N = self.N  # 16

        laynum = 21
        marray = [12544, 3136, 3136, 3136, 3136, 784, 784, 784, 784, 784, 196, 196, 196, 196, 196, 49, 49, 49, 49, 49, 1]
        karray = [147, 576, 576, 576, 576, 576, 1152, 64, 1152, 1152, 1152, 2304, 128, 2304, 2304, 2304, 4608, 256, 4608, 4608, 512]
        narray = [64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 1000]
        # narray = [16,  48,  48,  48,  32,  96, 113,  79,  96, 113, 210, 192, 207, 192, 225, 323, 481, 384, 420, 497, 650]


        # marray = [1.024e+03,1.024e+03,1.024e+03,1.024e+03,1.024e+03,1.024e+03,1.024e+03,2.560e+02,2.560e+02,1.280e+02,2.560e+02,2.560e+02,2.560e+02,2.560e+02,6.400e+01,6.400e+01,3.200e+01,6.400e+01,6.400e+01,6.400e+01,6.400e+01,1.000e+00]
        # karray = [27,144,144,144,144,144,144,144,288,32,288,288,288,288,288,576,64,576,576,576,576,64]
        # narray = [16,16,16,16,16,16,16,32,32,32, 32,32,32,32,64,64,64,64,64,64,64,10]

        # marray = [12544, 12544, 12544, 12544, 3136, 3136, 3136, 3136, 3136, 3136, 784, 784, 784, 784, 784, 784, 784, 784, 784, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 1]
        # karray = [27, 9, 32, 16, 9, 96, 24, 9, 144, 24, 9, 144, 32, 9, 192, 32, 9, 192, 32, 9, 192, 64, 9, 384, 64, 9, 384, 64, 9, 384, 64, 9, 384, 96, 9, 576, 96, 9, 576, 96, 9, 576, 160, 9, 960, 160, 9, 960, 160, 9, 960, 320, 1280]
        # # narray = [16, 32, 16, 96, 96, 24, 144, 144, 24, 144, 144, 32, 192, 192, 32, 192, 192, 32, 192, 192, 64, 384, 384, 64, 384, 384, 64, 384, 384, 64, 384, 384, 96, 576, 576, 96, 576, 576, 96, 576, 576, 160, 960, 960, 160, 960, 960, 160, 960, 960, 320, 1280, 1000]
        # narray = [16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16]


        m = marray[i]
        k = karray[i]
        n = round(narray[i] * ratio)
        
        if n == 0:
            return 0
        
        bits_l = Ab
        bits_R = Bb
        f = 64

        felog = 3
        rmem_num_regions = math.pow(2, felog)
        rmem_region_size = RMEM/rmem_num_regions
        tiles_m = math.ceil(m/M)
        tiles_n = math.ceil(n/N)
        tiles_k = math.ceil(k/K)
        lmem_region_size = tiles_k * bits_l
        lmem_num_regions = math.floor(LMEM/(lmem_region_size))
        lhs_fetches = math.floor(tiles_m/lmem_num_regions)
        if(tiles_m % lmem_num_regions != 0):
            lhs_fetches += 1
        last_iter_m = tiles_m % lmem_num_regions

        if(last_iter_m != 0):
            total_iters = lmem_num_regions * tiles_n * (lhs_fetches - 1) + last_iter_m * tiles_n
        else:
            total_iters = lmem_num_regions * tiles_n * lhs_fetches

        fetch_base = 32
        fetch_trans = 6 + 1 + 2
        lfetch_bram = M * (K*bits_l*tiles_k/f)
        # rfetch_bram = N * (K*bits_R*tiles_k/f)
        ## modify
        rfetch_bram = N * (tiles_k * bits_R * K/f)

        base_latency = 238
        sync_latency = 5


        popcount_latency = {'8':1, '16':1, '32':2, '64':3, '128':3, '256':4, '512':5}
        # Exce_latency = (bits_R * bits_l * tiles_k) * 5 + popcount_latency[str(K)] + 5
        # print(Exce_latency)

        Exce_latency = (tiles_k + 8) * bits_R * bits_l + 8


        lfetch_latency = (fetch_base + fetch_trans + lfetch_bram + sync_latency)
        rfetch_latency = (fetch_base + fetch_trans + rfetch_bram + sync_latency)

        if (lfetch_latency-1 > Exce_latency):
            delt = 0
        else:
            delt = 10 + (lmem_num_regions - 3) * (Exce_latency - lfetch_latency)

        if total_iters > lmem_num_regions + 3:
            if lhs_fetches > 1:
                fetch_receive = ((lmem_num_regions - 3) * (tiles_n - 1) * lhs_fetches + last_iter_m) * Exce_latency
            else:
                if last_iter_m > 3:
                    fetch_receive = (last_iter_m - 3) * (tiles_n - 1) * Exce_latency
                else:
                    fetch_receive = 0
        else:
            fetch_receive = 0

        fetch_receive = fetch_receive + delt * lhs_fetches

        lfetch_run_sync = (fetch_base + fetch_trans + lfetch_bram + sync_latency)*total_iters/tiles_n
        rfetch_run_sync = (fetch_base + fetch_trans + rfetch_bram + sync_latency)*lhs_fetches * tiles_n

        Latency = base_latency + lfetch_run_sync + rfetch_run_sync + fetch_receive
        return(Latency * pow(10, -5))

    def cal_dsp(self, i, ratio):
        if ratio == 1:
            return 0
        
        marray = [12544, 3136, 3136, 3136, 3136, 784, 784, 784, 784, 784, 196, 196, 196, 196, 196, 49, 49, 49, 49, 49, 1]
        karray = [147, 576, 576, 576, 576, 576, 1152, 64, 1152, 1152, 1152, 2304, 128, 2304, 2304, 2304, 4608, 256, 4608, 4608, 512]
        narray = [64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 1000]
        sarray = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]


        # BRAM depth
        Dl = self.Lmem  # 1024 * 2
        Dr = self.Rmem  # 1024 * 1

        Dr_Num = 2

        # configuration / setting
        Batch = 1
        InputChannel = 16
        OutputChannel = 16

        # input bit-width
        f = 64

        M = marray[i]
        K = karray[i]
        N = round(narray[i] * (1-ratio))
        S = sarray[i]

        ABits = 8
        WBits = 8

        # 8-bit
        if S == 1:
            tiles_k = math.ceil(K/InputChannel)
            tiles_m = math.ceil(M/Batch)
            tiles_n = math.ceil(N/OutputChannel)
        else:
        # 4-bit, half amount of bits, each DSP computes 2 data
            tiles_k = math.ceil(K / 2 / InputChannel)
            tiles_m = math.ceil(M / Batch)
            tiles_n = math.ceil(N / OutputChannel)
        region_l = math.floor(Dl/tiles_k)
        # region_r = math.floor(Dr/tiles_k/(OutputChannel/Dr_Num))
        region_r = math.floor(Dr / tiles_k / Dr_Num)
        tiles_region_l = math.floor(tiles_m/region_l)
        if tiles_m % region_l == 0:
            last_iters_l = 0
        else:
            last_iters_l = tiles_m % region_l

        tiles_region_r = math.floor(tiles_n/region_r)
        if tiles_n % region_r == 0:
            last_iters_r = 0
        else:
            last_iters_r = tiles_n % region_r


        # execLatency = tiles_k * region_r * (region_l + (OutputChannel / Dr_Num))
        basicLatency = 32
        # readLatencyR = basicLatency + OutputChannel + math.ceil(InputChannel * WBits / 32) + tiles_k * (InputChannel * WBits / 64)
        readLatencyR = basicLatency + OutputChannel/Dr_Num + tiles_k * region_r * (InputChannel * WBits / f) * (OutputChannel / Dr_Num)
        # readLatencyR = 4400
        # readLatencyL = basicLatency + Batch + tiles_k * 8 * (InputChannel * ABits / 64) + math.ceil(InputChannel * ABits / 32)
        readLatencyL = basicLatency + Batch + tiles_k * 8 * (InputChannel * ABits / f) * Batch

        execLatencyLite = tiles_k * region_r * Dr_Num
        readlatencyLite = tiles_k * (InputChannel * ABits / 64) * Batch

        # if execLatencyLite > readLatencyR:
        #     LatencyLite = execLatency + readLatencyL + readLatencyR
        # else:
        #     LatencyLite = execLatency * (readLatencyR - execLatencyLite) + readLatencyR + readLatencyL

        if execLatencyLite > readlatencyLite:
            syncLatency = 0
        else:
            syncLatency = readlatencyLite - execLatencyLite
            # print("syny")

        if last_iters_r != 0:
            execLatencyLiteLast = tiles_k * last_iters_r * Dr_Num
            if execLatencyLiteLast > readlatencyLite:
                syncLatencyLast = 0
            else:
                syncLatencyLast = readlatencyLite - execLatencyLiteLast
                # print("syny")
        else:
            syncLatencyLast = 0

        execLatency = tiles_k * region_r * region_l * Dr_Num * (1 + syncLatency) + 6
        # execLatency = tiles_k * region_r * (region_l + Dr_Num) * (1 + syncLatency) + 6



        # LatencyLiteL = tiles_k * (last_iters_l + (OutputChannel / Dr_Num)) * region_r + readLatencyR + basicLatency + Batch + tiles_k * (last_iters_l) * (InputChannel * ABits / 64) + math.ceil(InputChannel * ABits / 32)
        LatencyLiteL = tiles_k * last_iters_l * Dr_Num * region_r * (1 + syncLatency) + readLatencyR + basicLatency + Batch + tiles_k * 8 * (InputChannel * ABits / 64) * Batch

        # LatencyLiteR = tiles_k * (region_l + (OutputChannel / Dr_Num)) * last_iters_r + readLatencyL + basicLatency + OutputChannel + math.ceil(InputChannel * WBits / 32) + tiles_k * region_r
        LatencyLiteR = tiles_k * region_l * Dr_Num * last_iters_r * (1 + syncLatencyLast) + readLatencyL + basicLatency + OutputChannel + basicLatency + OutputChannel + tiles_k * last_iters_r * (InputChannel * WBits / 64) * (OutputChannel / Dr_Num)

        # LatencyLiteLR = tiles_k * last_iters_r * (last_iters_l + (OutputChannel / Dr_Num)) + readLatencyR + basicLatency + Batch + tiles_k * (last_iters_l) * (InputChannel * ABits / 64) + math.ceil(InputChannel * ABits / 32)
        LatencyLiteLR = tiles_k * last_iters_r * last_iters_l * Dr_Num * (1 + syncLatencyLast) + basicLatency + Batch + tiles_k * 8 * (InputChannel * ABits / 64) * Batch + basicLatency + OutputChannel + tiles_k * last_iters_r * (InputChannel * WBits / 64) * (OutputChannel / Dr_Num)

        Latency = tiles_region_l * tiles_region_r * (execLatency + readLatencyR + readLatencyL) + \
                        tiles_region_r * LatencyLiteL + \
                        tiles_region_l * LatencyLiteR + \
                        LatencyLiteLR + 32

        return(Latency * pow(10, -5))
    
    def cal_ratio(self, i, Ab, Bb):
        p = np.arange(0., 1.01, 0.01)
        for ratio in p: 
            if self.cal_dsp(i, ratio) < self.cal_bismo(i, ratio, Ab, Bb):
                if self.cal_bismo(i, ratio, Ab, Bb) > self.cal_dsp(i, ratio-0.01):
                    ratio = ratio-0.01
                    break
                else:
                    break
        return ratio

    def cal_lat(self, i, ratio, Ab, Bb):
        if self.cal_dsp(i, ratio) > self.cal_bismo(i, ratio, Ab, Bb):
            lat = self.cal_dsp(i, ratio)
        else:
            lat = self.cal_bismo(i, ratio, Ab, Bb)
        return lat
