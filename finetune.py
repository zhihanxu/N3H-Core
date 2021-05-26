# Code for "N3H-Core: Neuron-designed Neural Network Accelerator via FPGA-based Heterogeneous Computing Cores"

import os
import time
import math
import random
import shutil
import argparse
from numpy.core.numeric import False_

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.models as models
import customized_models

from lib.utils.utils import Logger, AverageMeter, accuracy
from lib.utils.data_utils import get_dataset
from progress.bar import Bar
from lib.utils.quantize_utils_dsp84 import QConv2d, QLinear, calibrate


# Models
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='data/imagenet', type=str)
parser.add_argument('--data_name', default='imagenet', type=str)
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--warmup_epoch', default=0, type=int, metavar='N',
                    help='manual warmup epoch number (useful on restarts)')
parser.add_argument('--train_batch', default=256, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test_batch', default=512, type=int, metavar='N',
                    help='test batchsize (default: 512)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_type', default='cos', type=str,
                    help='lr scheduler (exp/cos/step3/fixed)')
parser.add_argument('--schedule', type=int, nargs='+', default=[31, 61, 91],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', action='store_true',
                    help='use pretrained model')
# Quantization
parser.add_argument('--linear_quantization', dest='linear_quantization', action='store_true',
                    help='quantize both weight and activation)')
parser.add_argument('--free_high_bit', default=True, type=bool,
                    help='free the high bit (>6)')
parser.add_argument('--half', action='store_true',
                    help='half')
parser.add_argument('--half_type', default='O1', type=str,
                    help='half type: O0/O1/O2/O3')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50', choices=model_names,
                    help='model architecture:' + ' | '.join(model_names) + ' (default: resnet50)')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# Device options
parser.add_argument('--gpu_id', default='1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--customize', default=False, action='store_true')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
lr_current = state['lr']

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)


best_acc = 0  # best test accuracy


def load_my_state_dict(model, state_dict):
    model_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in model_state:
            continue
        param_data = param.data
        if model_state[name].shape == param_data.shape:
            # print("load%s"%name)
            model_state[name].copy_(param_data)


def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient
        optimizer.zero_grad()
        if args.half:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            # with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
        else:
            loss.backward()
        # do SGD step
        optimizer.step()

        if not args.linear_quantization:
            kmeans_update_model(model, quantizable_idx, centroid_label_dict, free_high_bit=args.free_high_bit)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if batch_idx % 1 == 0:
            bar.suffix = \
                '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
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
    return losses.avg, top1.avg


def test(val_loader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        # switch to evaluate mode
        model.eval()

        end = time.time()
        bar = Bar('Processing', max=len(val_loader))
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            if batch_idx % 1 == 0:
                bar.suffix  = \
                    '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                    'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
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
    return top1.avg, top5.avg


def save_checkpoint(state, is_best, checkpoint='checkpoints', filename='res18_image_441.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'res18_image_441_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global lr_current
    global best_acc
    if epoch < args.warmup_epoch:
        lr_current = state['lr']*args.gamma
    elif args.lr_type == 'cos':
        # cos
        lr_current = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
    elif args.lr_type == 'exp':
        step = 1
        decay = args.gamma
        lr_current = args.lr * (decay ** (epoch // step))
    elif epoch in args.schedule:
        lr_current *= args.gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_current


if __name__ == '__main__':
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    train_loader, val_loader, n_class = get_dataset(dataset_name=args.data_name, batch_size=args.train_batch,
                                                    n_worker=args.workers, data_root=args.data)
    if not args.customize:
        model = models.__dict__[args.arch](pretrained=args.pretrained)
    else:
        model = customized_models.__dict__[args.arch](pretrained=args.pretrained)

    print("=> creating model '{}'".format(args.arch), ' pretrained is ', args.pretrained)
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    cudnn.benchmark = True
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    quantizable_idx = []
    for i, m in enumerate(model.modules()):
        if type(m) in [QConv2d, QLinear]:
            quantizable_idx.append(i)
    # print(model)
    print('idx:', quantizable_idx)
    
    if args.evaluate is False:
        if 'mobilenetv2' in args.arch:
            strategy = [[8, -1], [6, 3], [8, 6], [6, 8], [6, 4], [5, 6], [6, 6], [6, 5], [5, 4], [8, 4], [4, 4], [4, 5], [5, 8], [3, 8], [4, 7], [5, 5], [4, 5], [4, 6], [5, 5], [2, 5], [8, 7], [3, 7], [3, 6], [5, 8], [7, 7], [2, 7], [2, 6], [4, 4], [3, 3], [2, 7], [4, 7], [2, 5], [2, 6], [3, 6], [2, 4], [2, 5], [3, 7], [3, 3], [2, 3], [3, 6], [2, 5], [3, 5], [2, 6], [2, 2], [3, 6], [2, 7], [7, 2], [2, 3], [4, 3], [5, 2], [2, 2], [4, 6], [8, 8]]
        elif 'res18_image' in args.arch:
            # strategy = [[8, 4, 1]] + [[4, 4, 1]] * 19 + [[4, 4, 1]]
            # strategy = [[8, 8, 0.13], [3, 5, 0.51], [2, 4, 0.64], [7, 2, 0.47], [2, 5, 0.61], [7, 5, 0.32], [2, 6, 0.73], [7, 6, 0], [5, 8, 0.46], [6, 8, 0.45], [4, 4, 0.68], [7, 5, 0.6], [7, 2, 0.1], [4, 8, 0.64], [3, 3, 0.82], [5, 7, 0.52], [4, 6, 0.75], [2, 8, 0.41], [7, 4, 0.71], [3, 5, 0.78], [8, 8, 0]] 
            # test
            # strategy = [[8, 8, 0.73], [4, 3, 0.52], [3, 2, 0.45], [2, 4, 0.46], [5, 3, 0.55], [3, 4, 0.51], [2, 3, 0.44], [4, 2, 0.49], [3, 3, 0.48], [2, 4, 0.46], [2, 2, 0.42],
            #             [2, 3, 0.44], [2, 2, 0.42], [4, 4, 0.55], [3, 2, 0.45], [4, 3, 0.52], [2, 4, 0.46], [3, 3, 0.48], [2, 4, 0.46], [5, 2, 0.52], [8, 8, 0.73]]
            # 0.6
            # strategy = [[8, 8, 0], [4, 4, 0.5], [4, 4, 0.5], [4, 4, 0.5], [4, 4, 0.5], [4, 4, 0.5], [4, 4, 0.66], [4, 4, 0.13], [4, 4, 0.66], [4, 4, 0.66], [4, 4, 0.94], [4, 4, 0.75], [4, 4, 0.13], [4, 4, 0.75], [4, 4, 0.75], [4, 4, 0.72], [4, 4, 0.85], [4, 4, 0.46], [4, 4, 0.85], [4, 4, 0.85], [8, 8, 0.28]]
            # # 0.5
            strategy = [[8, 8, 0.25], [3, 7, 0.5], [3, 7, 0.5], [3, 7, 0.5], [3, 7, 0.5], [2, 3, 0.75], [2, 3, 0.88], [4, 8, 0.13], [3, 3, 0.75], [2, 3, 0.75], [2, 3, 0.82], 
                        [3, 3, 0.81], [4, 7, 0.06], [3, 3, 0.81], [2, 5, 0.82], [2, 2, 0.85], [3, 2, 0.9], [4, 6, 0.05], [3, 3, 0.85], [2, 4, 0.82], [8, 8, 0.3]]
        elif 'resnet20' in args.arch:
            # strategy = [[8, 8, 1]] + [[4, 4, 1]] * 20 + [[8, 8, 1]]
            # strategy = [[8, 8, 0.03], [2, 2, 0.54], [2, 3, 0.54], [4, 3, 0.1], [4, 4, 0.03], [2, 2, 0.54], [3, 2, 0.35], [4, 2, 1.0], [2, 4, 1.0], [3, 3, 1.0], [2, 3, 1.0], [2, 4, 1.0], [3, 3, 0.51], [2, 4, 1.0], [2, 3, 1.0], [2, 3, 1.0], [2, 5, 0.0], [4, 3, 0.63], [2, 4, 1.0], [2, 4, 1.0], [2, 2, 1.0], [8, 8, 0.05]]
            strategy = [[8, 8, 0.03], [6, 2, 0.03], [6, 2, 0.03], [6, 2, 0.03], [6, 2, 0.03], [6, 2, 0.03], [6, 2, 0.03], [6, 2, 0.51], [4, 6, 0.51], [5, 2, 1.0], [4, 6, 0.51], [4, 5, 0.51], [4, 6, 0.51], [4, 6, 0.51], [5, 6, 0.64], [3, 4, 0.88], [7, 3, 1.0], [3, 4, 0.88], [3, 4, 0.88], [3, 4, 0.88], [3, 4, 0.88], [8, 8, 0.05]]
        else:
            raise NotImplementedError

        print(strategy)
        quantize_layer_bit_dict = {n: b for n, b in zip(quantizable_idx, strategy)}
        for i, layer in enumerate(model.modules()):
            if i not in quantizable_idx:
                continue
            else:
                layer.a_bit = quantize_layer_bit_dict[i][0]
                layer.w_bit = quantize_layer_bit_dict[i][1]
                layer.ratio = quantize_layer_bit_dict[i][2]
                      
        
        model = model.cuda()
        model = calibrate(model, train_loader)   


        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model = model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # Resume
    title = 'ImageNet-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)    #include activation range
        # strategy = [[8, 8, 0.51], [7, 7, 0.45], [6, 7, 0.41], [6, 7, 0.41], [6, 7, 0.41], [6, 7, 0.41], [6, 6, 0.37], [4, 5, 0.25], [4, 5, 0.25], [5, 8, 0.4], [4, 5, 0.25], 
        #             [4, 5, 0.25], [4, 5, 0.25], [4, 5, 0.25], [7, 7, 0.45], [5, 5, 0.3], [7, 6, 0.41], [5, 5, 0.3], [5, 5, 0.3], [5, 5, 0.3], [5, 4, 0.25], [8, 8, 0.51]]
        strategy = checkpoint['strategy']
        print(strategy)
        quantize_layer_bit_dict = {n: b for n, b in zip(quantizable_idx, strategy)}
        for i, layer in enumerate(model.modules()):
            if i not in quantizable_idx:
                continue
            else:
                layer.w_bit = quantize_layer_bit_dict[i][1]
                layer.a_bit = quantize_layer_bit_dict[i][0]
                layer.ratio = quantize_layer_bit_dict[i][2]
                
        model = model.cuda()
        model = calibrate(model, train_loader)
        model = torch.nn.DataParallel(model).cuda()
        
        best_acc = checkpoint['best_acc']
        print(best_acc)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        # model = calibrate(model, train_loader)
        # for name in checkpoint['state_dict'].keys():  # model.state_dict()
        #     print(name)
        print(model.state_dict()['module.fc.weight_range'])
        print(model.state_dict()['module.fc.activation_range'])
        # test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda)
        # print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        
        
        if os.path.isfile(os.path.join(args.checkpoint, 'log.txt')):
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
            logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    if args.evaluate:
        print('\nEvaluation only')
        top1_acc, top5_acc = test(val_loader, model, criterion, start_epoch, use_cuda)
        print(' Top1:  %.2f, Top5:  %.2f' % (top1_acc, top5_acc))
        exit()

    # test_acc = validate(criterion=nn.CrossEntropyLoss().cuda(), val_loader=val_loader, model=model, verbose=False)
    
    test_loss, test_acc = test(val_loader, model, criterion, 0, use_cuda)
    print('\ntest_acc: %f' %  (test_acc))
    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        
        print('Epoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr_current))

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        top1_acc, top5_acc = test(val_loader, model, criterion, epoch, use_cuda)
        print('\ntrain_acc: %f top1: %f top5: %f' % (train_acc, top1_acc, top5_acc))
        # append logger file
        logger.append([lr_current, train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = top1_acc > best_acc
        best_acc = max(top1_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': top1_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
                'strategy': strategy,
            }, is_best, checkpoint=args.checkpoint)

    logger.close()

    print('Best acc:')
    print(best_acc)

