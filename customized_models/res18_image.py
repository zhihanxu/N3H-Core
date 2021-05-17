import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from torch.utils.data.sampler import SubsetRandomSampler
from lib.utils.quantize_utils_ru import QConv2d, QLinear, QModule, calibrate   
import time
import torchvision.datasets as datasets
from lib.utils.data_utils import get_dataset, get_split_train_dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
__all__ = ['ResNet', 'res18_image']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return QConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return QConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv1x1_down(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return QConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=
                 100, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = QConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False, half_wave=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = QLinear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, QConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1_down(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = torch.load('./pretrained/resnet18-5c106cde.pth')
        state_dict['fc.weight'] = state_dict['fc.weight'][0:100, :]
        state_dict['fc.bias'] = state_dict['fc.bias'][0:100]
        model.load_state_dict(state_dict, strict=False)
    return model


def res18_image(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('res18_image', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def validate(criterion, val_loader, model, verbose=False):
    import time
    from progress.bar import Bar
    from lib.utils.utils import AverageMeter, accuracy, prGreen, measure_model
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

            input_var, target_var = inputs.to(device), targets.to(device)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

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
    return top1.avg

def finetune(criterion, train_loader, model, epochs=1, verbose=True):
    import time
    import torch.optim as optim
    from progress.bar import Bar
    from lib.utils.utils import AverageMeter, accuracy, prGreen, measure_model
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
            input_var, target_var = inputs.cuda(), targets.cuda()

            # measure data loading time
            data_time.update(time.time() - end)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            optimizer=optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
            # compute gradient
            optimizer.zero_grad()
            loss.backward()

            # do SGD step
            optimizer.step()

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

        if top1.avg > best_acc:
            best_acc = top1.avg

        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.8

    t2 = time.time()
    if verbose:
        print('* Test loss: %.3f  top1: %.3f  top5: %.3f  time: %.3f' % (losses.avg, top1.avg, top5.avg, t2-t1))
    return best_acc

def run_program():
    model = res18_image(pretrained=True).to(device)

    for name, module in model.named_modules():
        if isinstance(module, QModule):
            module.w_bit = 8
            module.a_bit = 8
            module.ratio = 1
    # strategy = [[4, -1]] + [[4, -1]]*20
    # quantizable_idx = []
    # for i, m in enumerate(model.modules()):
    #     if type(m) in [QConv2d, QLinear]:
    #         quantizable_idx.append(i)
            
    # quantize_layer_bit_dict = {n: b for n, b in zip(quantizable_idx, strategy)}
    # for i, layer in enumerate(model.modules()):
    #     if i not in quantizable_idx:
    #         continue
    #     else:
    #         layer.w_bit = quantize_layer_bit_dict[i][0]
    #         layer.a_bit = quantize_layer_bit_dict[i][1]
            
    # train_loader, val_loader, n_class = get_split_train_dataset(
    #     'imagenet', batch_size=128, n_worker=32, data_root='data/imagenet-data',
    #     val_size=10000, train_size=20000)

    # train_loader, val_loader, n_class = get_dataset(dataset_name='imagenet100', batch_size=128,
    #                                             n_worker=32, data_root='data/imagenet100')

    train_loader, val_loader, n_class = get_dataset(dataset_name='ILSVRC2012', batch_size=256,
                                                n_worker=32, data_root='data/ILSVRC2012')

    model = calibrate(model, val_loader)
    # model = torch.nn.DataParallel(model).cuda()
    acc = validate(criterion=nn.CrossEntropyLoss().cuda(), val_loader=val_loader, model=model, verbose=False) 
    # acc = finetune(criterion=nn.CrossEntropyLoss().cuda(), train_loader=train_loader, model=model, epochs=1)
    print(acc)
    return acc 

if __name__ == '__main__':
    episode = 0
    episode_num = 1
    all_acc = 0
    avg_acc = 0
    t1 = time.time()
    while episode < episode_num:
        all_acc += run_program()
        episode += 1
    t2 = time.time()
    avg_acc = all_acc/episode_num
    print('* Avg_acc: %.3f  time: %.3f' % (avg_acc, t2-t1))

