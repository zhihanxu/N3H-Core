# Code for "[HAQ: Hardware-Aware Automated Quantization with Mixed Precision"
# Kuan Wang*, Zhijian Liu*, Yujun Lin*, Ji Lin, Song Han
# {kuanwang, zhijian, yujunlin, jilin, songhan}@mit.edu

import math
from types import new_class
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _single, _pair, _triple

class QModule(nn.Module):
    def __init__(self, w_bit=-1, a_bit=-1, ratio=0.8, half_wave=True):
        super(QModule, self).__init__()
        
        self.ratio = ratio
        if half_wave:
            self._a_bit = a_bit
        else:
            self._a_bit = a_bit - 1
        self._w_bit = w_bit
        self._b_bit = 32
        self._half_wave = half_wave
        
        self.init_range = 6.
        self.activation_range = nn.Parameter(torch.Tensor([self.init_range]))
        self.weight_range = None
        self.channel_type = None

        self._quantized = True
        self._tanh_weight = False
        self._fix_weight = False
        self._trainable_activation_range = False
        self._calibrate = False

    @property
    def w_bit(self):
        return self._w_bit

    @w_bit.setter
    def w_bit(self, w_bit):
        self._w_bit = w_bit

    @property
    def a_bit(self):
        return self._a_bit

    @a_bit.setter
    def a_bit(self, a_bit):
        self._a_bit = a_bit
 
    @property
    def b_bit(self):
        return self._b_bit
        
    @property
    def half_wave(self):
        return self._half_wave
    
    @property
    def quantized(self):
        return self._quantized

    @property
    def tanh_weight(self):
        return self._tanh_weight

    def set_quantize(self, quantized):
        self._quantized = quantized

    def set_tanh_weight(self, tanh_weight):
        self._tanh_weight = tanh_weight
        if self._tanh_weight:
            self.weight_range.data[0] = 1.0

    def set_fix_weight(self, fix_weight):
        self._fix_weight = fix_weight

    def set_activation_range(self, activation_range):
        self.activation_range.data[0] = activation_range

    def set_weight_range(self, weight_range):
        self.weight_range.data[0] = weight_range

    def set_trainable_activation_range(self, trainable_activation_range=True):
        self._trainable_activation_range = trainable_activation_range
        self.activation_range.requires_grad_(trainable_activation_range)

    def set_calibrate(self, calibrate=True):
        self._calibrate = calibrate

    def set_tanh(self, tanh=True):
        self._tanh_weight = tanh

    def _compute_threshold(self, data, bitwidth):
        mn = 0
        mx = np.abs(data).max()
        if np.isclose(mx, 0.0):
            return 0.0
        hist, bin_edges = np.histogram(np.abs(data), bins='sqrt', range=(mn, mx), density=True)
        hist = hist / np.sum(hist)
        cumsum = np.cumsum(hist)
        n = pow(2, int(bitwidth) - 1)
        threshold = []
        scaling_factor = []
        d = []
        if n + 1 > len(bin_edges) - 1:
            th_layer_out = bin_edges[-1]
            # sf_layer_out = th_layer_out / (pow(2, bitwidth - 1) - 1)
            return float(th_layer_out)
        for i in range(n + 1, len(bin_edges), 1):
            threshold_tmp = (i + 0.5) * (bin_edges[1] - bin_edges[0])
            threshold = np.concatenate((threshold, [threshold_tmp]))
            scaling_factor_tmp = threshold_tmp / (pow(2, bitwidth - 1) - 1)
            scaling_factor = np.concatenate((scaling_factor, [scaling_factor_tmp]))
            p = np.copy(cumsum)
            p[(i - 1):] = 1
            x = np.linspace(0.0, 1.0, n)
            xp = np.linspace(0.0, 1.0, i)
            fp = p[:i]
            p_interp = np.interp(x, xp, fp)
            x = np.linspace(0.0, 1.0, i)
            xp = np.linspace(0.0, 1.0, n)
            fp = p_interp
            q_interp = np.interp(x, xp, fp)
            q = np.copy(p)
            q[:i] = q_interp
            d_tmp = np.sum((cumsum - q) * np.log2(cumsum / q))  # Kullback-Leibler-J
            d = np.concatenate((d, [d_tmp]))

        th_layer_out = threshold[np.argmin(d)]
        # sf_layer_out = scaling_factor[np.argmin(d)]
        threshold = float(th_layer_out)
        return threshold

    def _quantize_activation(self, inputs):
        # print("act bits = {}".format(self.a_bit))
        if self._quantized and self._a_bit > 0:
            if self._calibrate:
                # if self._a_bit < 5:
                threshold = self._compute_threshold(inputs.data.cpu().numpy(), self._a_bit)
                estimate_activation_range = min(min(self.init_range, inputs.abs().max().item()), threshold)
                # else:
                #     estimate_activation_range = min(self.init_range, inputs.abs().max().item())
                # print('range:', estimate_activation_range, '  shape:', inputs.shape, '  inp_abs_max:', inputs.abs().max())
                self.activation_range.data = torch.tensor([estimate_activation_range], device=inputs.device)
                return inputs

            if self._trainable_activation_range:
                if self._half_wave:
                    ori_x = 0.5 * (inputs.abs() - (inputs - self.activation_range).abs() + self.activation_range)
                else:
                    ori_x = 0.5 * ((-inputs - self.activation_range).abs() - (inputs - self.activation_range).abs())
            else:
                if self._half_wave:
                    ori_x = inputs.clamp(0.0, self.activation_range.item())
                else:
                    ori_x = inputs.clamp(-self.activation_range.item(), self.activation_range.item())

            scaling_factor = self.activation_range.item() / (2. ** self._a_bit - 1.)
            x = ori_x.detach().clone()
            x.div_(scaling_factor).round_().mul_(scaling_factor)

            # STE
            # x = ori_x + x.detach() - ori_x.detach()
            return STE.apply(ori_x, x)
        else:
            return inputs

    def _quantize_weight(self, weight):
        # print("wei bits = {}".format(self.a_bit))
        if self._tanh_weight:
            weight = weight.tanh()
            weight = weight / weight.abs().max()
        
        if self._quantized and self._w_bit > 0:   
            if self.weight_range is None:
                self.weight_range = nn.Parameter(torch.Tensor(list(channel.abs().max().item() for channel in weight)), requires_grad=False)
            
            dsp_number = math.floor(weight.shape[0] * self.ratio)
            bismo_number = weight.shape[0] - dsp_number
            
            if self.channel_type is None:
                self.channel_type = [0] * dsp_number + [1] * bismo_number    # 0 dsp, 1 bismo   
            
            if self._calibrate:
                # print(self.channel_type)
                self.channel_type = nn.Parameter(torch.Tensor(self.channel_type), requires_grad=False)
                
                for (idx, channel) in enumerate(weight):
                    # if self._w_bit < 5 and self.channel_type[idx] == 1:
                    threshold = self._compute_threshold(channel.data.cpu().numpy(), self._w_bit)
                    # else:
                    #     threshold = channel.abs().max().item()
                    self.weight_range[idx] = threshold
                return weight

            ori_w = weight
            new_w = ori_w.clone().detach()
            
            for (idx, channel) in enumerate(weight):            
                if self.channel_type[idx] == 0:
                    temp_w_bit = 8  #DSP 8bit
                else:
                    temp_w_bit = self._w_bit    #Bismo

                ori_ch = channel
                threshold = self.weight_range[idx]

                scaling_factor = threshold / (pow(2., temp_w_bit - 1) - 1.)
                new_ch = ori_ch.clamp(-threshold, threshold)
                # w[w.abs() > threshold - threshold / 64.] = 0.
                new_ch.div_(scaling_factor).round_().mul_(scaling_factor)
                new_w[idx] = new_ch
                
            # print("....... {} .....".format((new_ch == ori_ch).all()))  #false 23
            # print("******* {} *****".format((new_w == ori_w).all()))    #23 false for before each training batch
            
            # STE
            if self._fix_weight:
                # w = w.detach()
                return new_w.detach()
            else:
                # w = ori_w + w.detach() - ori_w.detach()
                return STE.apply(ori_w, new_w)  
                
        else:
            return weight   #same as original weight

    def _quantize_bias(self, bias):
        if bias is not None and self._quantized and self._b_bit > 0:
            if self._calibrate:
                return bias
            ori_b = bias
            threshold = ori_b.data.max().item() + 0.00001
            scaling_factor = threshold / (pow(2., self._b_bit - 1) - 1.)
            b = torch.clamp(ori_b.data, -threshold, threshold)
            b.div_(scaling_factor).round_().mul_(scaling_factor)
            # STE
            if self._fix_weight:
                return b.detach()
            else:
                # b = ori_b + b.detach() - ori_b.detach()
                return STE.apply(ori_b, b)
        else:
            return bias

    def _quantize(self, inputs, weight, bias):
        inputs = self._quantize_activation(inputs=inputs)
        weight = self._quantize_weight(weight=weight)
        # bias = self._quantize_bias(bias=bias)
        return inputs, weight, bias

    def forward(self, *inputs):
        raise NotImplementedError

    def extra_repr(self):
        return 'w_bit={}, a_bit={}, tanh_weight={}'.format(
            self.w_bit if self.w_bit > 0 else -1, self.a_bit if self.a_bit > 0 else -1, self._tanh_weight
        )


class STE(torch.autograd.Function):
    # for faster inference
    @staticmethod
    def forward(ctx, origin_inputs, wanted_inputs):
        return wanted_inputs.detach()

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None


class QConv2d(QModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 w_bit=-1, a_bit=-1, half_wave=True):
        super(QConv2d, self).__init__(w_bit=w_bit, a_bit=a_bit, half_wave=half_wave)
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):
        inputs, weight, bias = self._quantize(inputs=inputs, weight=self.weight, bias=self.bias)
        # print("+++++++++++++++++ {} ++++++++++++++++".format((weight == self.weight).all())) 20
        return F.conv2d(inputs, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.w_bit > 0 or self.a_bit > 0:
            s += ', w_bit={}, a_bit={}'.format(self.w_bit, self.a_bit)
            s += ', half wave' if self.half_wave else ', full wave'
        return s.format(**self.__dict__)


class QLinear(QModule):
    def __init__(self, in_features, out_features, bias=True, w_bit=-1, a_bit=-1, half_wave=True):
        super(QLinear, self).__init__(w_bit=w_bit, a_bit=a_bit, half_wave=half_wave)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, inputs):
        inputs, weight, bias = self._quantize(inputs=inputs, weight=self.weight, bias=self.bias)
        # print("---------------- {} -------------".format((weight == self.weight).all())) 3
        return F.linear(inputs, weight=weight, bias=bias)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)
        if self.w_bit > 0 or self.a_bit > 0:
            s += ', w_bit={w_bit}, a_bit={a_bit}'.format(w_bit=self.w_bit, a_bit=self.a_bit)
            s += ', half wave' if self.half_wave else ', full wave'
        return s


def calibrate(model, loader):
    data_parallel_flag = False
    if hasattr(model, 'module'):
        data_parallel_flag = True
        model = model.module
    print('==> start calibrate')
    for name, module in model.named_modules():
        if isinstance(module, QModule):
            module.set_calibrate(calibrate=True)
            # print('calibrate test')
    inputs, _ = next(iter(loader))
    # use 1 gpu to calibrate
    inputs = inputs.to('cuda:0', non_blocking=True)

    with torch.no_grad():
        model(inputs)
    for name, module in model.named_modules():
        if isinstance(module, QModule):
            module.set_calibrate(calibrate=False)
    print('==> end calibrate')
    if data_parallel_flag:
        model = nn.DataParallel(model)
    return model


def set_fix_weight(model, fix_weight=True):
    if fix_weight:
        print('\n==> set weight fixed')
    for name, module in model.named_modules():
        if isinstance(module, QModule):
            module.set_fix_weight(fix_weight=fix_weight)


