import sys
import os
import numpy as np
from collections import namedtuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function

QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])

_DEFAULT_FLATTEN = (1, -1)
_DEFAULT_FLATTEN_GRAD = (0, -1)


def _deflatten_as(x, x_full):
    shape = list(x.shape) + [1] * (x_full.dim() - x.dim())
    return x.view(*shape)


def calculate_qparams(x, num_bits, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0,  reduce_type='mean', keepdim=False, true_zero=False):
    with torch.no_grad():
        x_flat = x.flatten(*flatten_dims)
        if x_flat.dim() == 1:
            min_values = _deflatten_as(x_flat.min(), x)
            max_values = _deflatten_as(x_flat.max(), x)
        else:
            min_values = _deflatten_as(x_flat.min(-1)[0], x)
            max_values = _deflatten_as(x_flat.max(-1)[0], x)

        # reduce_dim: 0 -> layerwise, None -> channel wise for weight, image wise for activation
        # reduce_dim: "mean" -> mean value of each image in the batch, "extreme" -> the extreme value in the whole batch
        if reduce_dim is not None:
            if reduce_type == 'mean':
                min_values = min_values.mean(reduce_dim, keepdim=keepdim)
                max_values = max_values.mean(reduce_dim, keepdim=keepdim)
            else:
                min_values = min_values.min(reduce_dim, keepdim=keepdim)[0]
                max_values = max_values.max(reduce_dim, keepdim=keepdim)[0]
        # TODO: re-add true zero computation
        range_values = max_values - min_values
        return QParams(range=range_values, zero_point=min_values,
                       num_bits=num_bits)


def calculate_qparams_dws(x, num_bits):
    with torch.no_grad():
        min_values = x.min(-1)[0].min(-1)[0].min(0)[0].view(1, -1, 1, 1)
        max_values = x.max(-1)[0].max(-1)[0].max(0)[0].view(1, -1, 1, 1)

        range_values = max_values - min_values
        return QParams(range=range_values, zero_point=min_values,
                       num_bits=num_bits)




class my_clamp_round(InplaceFunction):

    @staticmethod
    def forward(ctx, input, min_value, max_value):
        ctx.input = input
        ctx.min = min_value
        ctx.max = max_value
        return torch.clamp(torch.round(input), min_value, max_value)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        mask = (ctx.input > ctx.min) * (ctx.input < ctx.max)
        grad_input = mask.float() * grad_input
        return grad_input, None, None


class UniformQuantize_STE(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN,
                reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False, beta=None, mode='soft', act_num=1, beta_param=None):

        output = input.clone()
        active_bit_list = None
        if qparams is None:
            assert num_bits is not None, "either provide qparams of num_bits to quantize"
            qparams = calculate_qparams(
                input, num_bits=num_bits, flatten_dims=flatten_dims, reduce_dim=reduce_dim)

        zero_point = qparams.zero_point
        num_bits = qparams.num_bits


        if type(num_bits) == int:
            qmin = 0.
            qmax = 2.**num_bits

            scale = qparams.range / (qmax - qmin)

            output.add_(qmin * scale - zero_point).div_(scale)

            output = my_clamp_round().apply(output, 0, int(2.**num_bits - 1.))

            output.mul_(scale).add_(zero_point - qmin * scale)  # dequantize

            active_bit_list = 0


        return output, active_bit_list


    @staticmethod
    def backward(ctx, *grad_output):
        # straight-through estimator
        grad_input = grad_output[0]
        return grad_input, None, None, None, None, None, None, None, None, None, None, None, None



class UniformQuantize():

    @staticmethod
    def quantize(input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN,
                reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False, beta=None, mode='soft', act_num=1, beta_param=None):

        output = input.clone()
        active_bit_list = None
        if qparams is None:
            assert num_bits is not None, "either provide qparams of num_bits to quantize"
            qparams = calculate_qparams(
                input, num_bits=num_bits, flatten_dims=flatten_dims, reduce_dim=reduce_dim)

        zero_point = qparams.zero_point
        num_bits = qparams.num_bits

        if type(num_bits) != int:
            assert type(num_bits) == list
            ######only for work
            beta = [1,1,1]

            assert beta is not None
            assert len(beta) == len(num_bits), 'len(beta):%d len(num_bits):%d' % (len(beta), len(num_bits))

            if len(num_bits) == 1:
                num_bits = num_bits[0]

        if type(num_bits) == int:
            qmin = 0.
            qmax = 2.**num_bits

            scale = qparams.range / (qmax - qmin)

            output.add_(qmin * scale - zero_point).div_(scale)

            output = my_clamp_round().apply(output, 0, int(2.**num_bits - 1.))

            output.mul_(scale).add_(zero_point - qmin * scale)  # dequantize

            active_bit_list = 0
            
        return output,scale


class UniformQuantize_Channel():

    @staticmethod
    def quantize(input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN,
                reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False, beta=None, mode='soft', act_num=1, beta_param=None):

        output = input.clone()
        active_bit_list = None
        if qparams is None:
            assert num_bits is not None, "either provide qparams of num_bits to quantize"
            qparams = calculate_qparams(
                input, num_bits=num_bits, flatten_dims=flatten_dims, reduce_dim=reduce_dim)

        zero_point = qparams.zero_point
        num_bits = qparams.num_bits

        if type(num_bits) != int:
            assert type(num_bits) == list
            ######only for work
            beta = [1,1,1]

            assert beta is not None
            assert len(beta) == len(num_bits), 'len(beta):%d len(num_bits):%d' % (len(beta), len(num_bits))

            if len(num_bits) == 1:
                num_bits = num_bits[0]

        if type(num_bits) == int:
            qmin = 0.
            qmax = 2.**num_bits

            scale = qparams.range / (qmax - qmin)

            output.add_(qmin * scale - zero_point).div_(scale)

            output = my_clamp_round().apply(output, 0, int(2.**num_bits - 1.))

            output.mul_(scale).add_(zero_point - qmin * scale)  # dequantize

            active_bit_list = 0
            
        return output, active_bit_list




def Quantize(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False, beta=None, mode='soft', act_num=1, beta_param=None):
    return UniformQuantize.quantize(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic, inplace, beta, mode, act_num, beta_param)


def Quantize_STE(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False, beta=None, mode='soft', act_num=1, beta_param=None):
    return UniformQuantize_STE.apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic, inplace, beta, mode, act_num, beta_param)



class QConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False, dws=False, hetero=False):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)

        self.dws = dws

        self.active_bit_list = None

        self.momentum = 0.1

        if self.dws:
            shape_measure = (1, in_channels, 1, 1)
        else:
            shape_measure = (1, 1, 1, 1)

        self.register_buffer('running_zero_point', torch.zeros(*shape_measure))
        self.register_buffer('running_range', torch.zeros(*shape_measure))

        self.stage = 'update_arch'

        self.hetero = hetero


    def forward(self, input, num_bits=32, beta=None, mode='soft', act_num=None, beta_param=None):
        # print(num_bits)
        # print(act_num)
        if num_bits == 32:
            output = F.conv2d(input, self.weight, self.bias, self.stride,
                                  self.padding, self.dilation, self.groups)
        else:
            if self.stage == 'update_arch':
                quantize_fn = Quantize
            elif self.stage == 'update_weight':
                quantize_fn = Quantize_STE
                
                if self.hetero:
                    mode = 'soft'
            else:
                print('Wrong stage:', self.stage)
                sys.exit()

            if self.training:
                if self.dws:
                    qparams = calculate_qparams_dws(input, num_bits=num_bits)
                else:
                    qparams = calculate_qparams(input, num_bits=num_bits, flatten_dims=(1, -1), reduce_dim=0, reduce_type='extreme')
                with torch.no_grad():
                    self.running_zero_point.mul_(self.momentum).add_(
                        qparams.zero_point * (1 - self.momentum))
                    self.running_range.mul_(self.momentum).add_(
                        qparams.range * (1 - self.momentum))
            else:
                qparams = QParams(range=self.running_range,
                  zero_point=self.running_zero_point, num_bits=num_bits)

            qinput, active_bit_list = quantize_fn(input, qparams=qparams, dequantize=True,
                               stochastic=False, inplace=False, beta=beta, mode=mode, act_num=act_num, beta_param=beta_param)

            self.set_active_bit_list(active_bit_list)

            weight_qparams = calculate_qparams(
                self.weight, num_bits=num_bits, flatten_dims=(1, -1), reduce_dim=None)
            qweight, _ = quantize_fn(self.weight, qparams=weight_qparams, beta=beta, mode=mode, act_num=act_num, beta_param=beta_param)

            if self.bias is not None:
                qbias, _ = quantize_fn(
                    self.bias, num_bits=num_bits,
                    flatten_dims=(0, -1))
            else:
                qbias = None
            
            output = F.conv2d(qinput, qweight, qbias, self.stride,
                                  self.padding, self.dilation, self.groups)

        return output


    def set_active_bit_list(self, active_bit_list):
        self.active_bit_list = active_bit_list


    def set_stage(self, stage):
        assert stage == 'update_weight' or stage == 'update_arch'
        self.stage = stage


class QLinear(nn.Linear):
    """docstring for QConv2d."""

    def __init__(self, in_features, out_features, bias=True, hetero=False):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.momentum = 0.1

        shape_measure = (1,)
        self.register_buffer('running_zero_point', torch.zeros(*shape_measure))
        self.register_buffer('running_range', torch.zeros(*shape_measure))

        self.stage = 'update_weight'

        self.hetero = hetero


    def forward(self, input, num_bits=32, beta=None, mode='soft'):
        if num_bits < 32:
            if self.stage == 'update_arch':
                quantize_fn = Quantize
            elif self.stage == 'update_weight':
                quantize_fn = Quantize_STE

                if self.hetero:
                    mode = 'soft'
            else:
                print('Wrong stage:', self.stage)
                sys.exit()

            if self.training:
                qparams = calculate_qparams(
                        input, num_bits=num_bits, flatten_dims=(1, -1), reduce_dim=0, reduce_type='extreme')
                with torch.no_grad():
                    self.running_zero_point.mul_(self.momentum).add_(
                        qparams.zero_point * (1 - self.momentum))
                    self.running_range.mul_(self.momentum).add_(
                        qparams.range * (1 - self.momentum))
            else:
                qparams = QParams(range=self.running_range,
                  zero_point=self.running_zero_point, num_bits=num_bits)

            qinput, active_bit_list = quantize_fn(input, qparams=qparams, dequantize=True,
                               stochastic=False, inplace=False, beta=beta, mode=mode)

            self.set_active_bit_list(active_bit_list)

            weight_qparams = calculate_qparams(
                self.weight, num_bits=num_bits, flatten_dims=(1, -1), reduce_dim=None)
            qweight, _ = quantize_fn(self.weight, qparams=weight_qparams, beta=beta, mode=mode)

            if self.bias is not None:
                qbias, _ = quantize_fn(
                    self.bias, num_bits=num_bits,
                    flatten_dims=(0, -1))
            else:
                qbias = None

            output = F.linear(qinput, qweight, qbias)

        else:
            output = F.linear(input, self.weight, self.bias)

        return output


    def set_active_bit_list(self, active_bit_list):
        self.active_bit_list = active_bit_list


    def set_stage(self, stage):
        assert stage == 'update_weight' or stage == 'update_arch'
        self.stage = stage


if __name__ == '__main__':
    # x = torch.rand(2, 3)
    # x_q = Quantize(x, flatten_dims=(-1), num_bits=8, dequantize=True)
    # print(x)
    # print(x_q)

    conv = QConv2d(3, 8, 1)
    num_bits = [6, 7, 8]
    beta = nn.Parameter(torch.autograd.Variable(torch.ones(3)))

    output = conv(torch.rand(1, 3, 16, 16), num_bits=num_bits, beta=beta)
    # output = conv(torch.rand(1, 3, 16, 16), num_bits=num_bits, beta=beta)
    # print(output)