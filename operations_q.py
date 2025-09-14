from pdb import set_trace as bp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from thop import profile
# from thop.count_hooks import count_convNd
from quantize_tile import MixActivConv2d
import sys
import os.path as osp
import time
from config_train_super_q import config

__all__ = ['ConvBlock', 'Skip','ConvNorm', 'OPS']

quant_bit = config.num_bits_list #[2,4,8] [2,3,4,5,6,7,8]
Conv2d = MixActivConv2d
BatchNorm2d = nn.BatchNorm2d

op_list = []

def convert_to_binary(value, options):
    return [1 if value == opt else 0 for opt in options]

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        assert (
            C % g == 0
        ), "Incompatible group size {} for input channel {}".format(g, C)
        return (
            x.view(N, g, int(C / g), H, W)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(N, C, H, W)
        )


class ConvBlock_N(nn.Module):
    '''
    conv => norm => activation
    use native Conv2d, not slimmable
    '''
    def __init__(self, C_in, C_out,  layer_id, expansion=1, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        super(ConvBlock_N, self).__init__()
        self.C_in = C_in
        self.C_out = C_out

        self.layer_id = layer_id

        # self.register_buffer('active_bit_list', torch.tensor([0]).cuda())

        assert type(expansion) == int
        self.expansion = expansion
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        if padding is None:
            # assume h_out = h_in / s
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.))
        else:
            self.padding = padding
        self.dilation = dilation
        assert type(groups) == int
        self.groups = groups
        self.bias = bias

        if self.groups > 1:
            self.shuffle = ChannelShuffle(self.groups)

        self.conv1 = nn.Conv2d(C_in, C_in*expansion, kernel_size=1, stride=1, padding=0, dilation=1, groups=self.groups, bias=bias)
        self.bn1 = BatchNorm2d(C_in*expansion)

        self.conv2 = nn.Conv2d(C_in*expansion, C_in*expansion, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=1, groups=C_in*expansion, bias=bias)
        self.bn2 = BatchNorm2d(C_in*expansion)


        self.conv3 = nn.Conv2d(C_in*expansion, C_out, kernel_size=1, stride=1, padding=0, dilation=1, groups=self.groups, bias=bias)
        self.bn3 = BatchNorm2d(C_out)

        self.nl = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = x
        x = self.nl(self.bn1(self.conv1(x)))

        if self.groups > 1:
            x = self.shuffle(x)

        x = self.nl(self.bn2(self.conv2(x)))

        x = self.bn3(self.conv3(x))

        if self.C_in == self.C_out and self.stride == 1:
            x += identity

        return x
    

class ConvBlock(nn.Module):
    '''
    conv => norm => activation
    use native Conv2d, not slimmable
    '''
    def __init__(self, C_in, C_out,  layer_id, expansion=1, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False,gpu = 0):
        super(ConvBlock, self).__init__()
        self.C_in = C_in
        self.C_out = C_out

        self.layer_id = layer_id

        # self.register_buffer('active_bit_list', torch.tensor([0]).cuda())

        assert type(expansion) == int
        self.expansion = expansion
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        if padding is None:
            # assume h_out = h_in / s
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.))
        else:
            self.padding = padding
        self.dilation = dilation
        assert type(groups) == int
        self.groups = groups
        self.bias = bias

        if self.groups > 1:
            self.shuffle = ChannelShuffle(self.groups)

        self.conv1 = Conv2d(C_in, C_in*expansion, wbits = quant_bit , abits = quant_bit , share_weight = False , kernel_size=1, stride=1, padding=0, dilation=1, groups=self.groups, bias=bias,gpu = gpu)
        self.bn1 = BatchNorm2d(C_in*expansion)

        self.conv2 = Conv2d(C_in*expansion, C_in*expansion,wbits = quant_bit , abits = quant_bit , share_weight = False ,  kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=1, groups=C_in*expansion, bias=bias,gpu = gpu)
        self.bn2 = BatchNorm2d(C_in*expansion)


        self.conv3 = Conv2d(C_in*expansion, C_out, wbits = quant_bit , abits = quant_bit , share_weight = False , kernel_size=1, stride=1, padding=0, dilation=1, groups=self.groups, bias=bias,gpu = gpu)
        self.bn3 = BatchNorm2d(C_out)

        self.nl = nn.ReLU(inplace=True)
        self.scale = []

    def forward(self, x, beta=None, quant_choose = 0):
        self.scale = []
        identity = x
        x,scale = self.conv1(x,beta=beta,quant_choose = quant_choose)
        x = self.nl(self.bn1(x))
        self.scale.append(scale)
        # op_list.append([1,1,x.size()[2],self.C_in,self.C_in*self.expansion])
        # a = convert_to_binary(1, [1, 3, 5])
        # b = convert_to_binary(1, [1, 2])
        # c = convert_to_binary(x.size()[2], [32, 16, 8, 4])
        # d = convert_to_binary(self.C_in*self.expansion, [16,24,32,48,64,96,112,144,184,192,336,352,384,552,672,1104])
        # e = convert_to_binary(self.C_out, [16,24,32,48,64,96,112,144,184,192,336,352,384,552,672,1104])
        # op_list.append([a,b,c,d,e])

        if self.groups > 1:
            x = self.shuffle(x)

        x,scale = self.conv2(x,beta=beta,quant_choose = quant_choose)
        x = self.nl(self.bn2(x))
        self.scale.append(scale)
        # op_list.append([self.kernel_size,self.stride,x.size()[2],self.C_in*self.expansion,self.C_in*self.expansion])
        # a = convert_to_binary(1, [1, 3, 5])
        # b = convert_to_binary(1, [1, 2])
        # c = convert_to_binary(x.size()[2], [32, 16, 8, 4])
        # d = convert_to_binary(self.C_in*self.expansion, [16,24,32,48,64,96,112,144,184,192,336,352,384,552,672,1104])
        # e = convert_to_binary(self.C_out, [16,24,32,48,64,96,112,144,184,192,336,352,384,552,672,1104])
        # op_list.append([a,b,c,d,e])

        x,scale = self.conv3(x,beta=beta,quant_choose = quant_choose)
        x = self.bn3(x)
        self.scale.append(scale)
        # op_list.append([1,1,x.size()[2],self.C_in*self.expansion,self.C_out])
        # a = convert_to_binary(1, [1, 3, 5])
        # b = convert_to_binary(1, [1, 2])
        # c = convert_to_binary(x.size()[2], [32, 16, 8, 4])
        # d = convert_to_binary(self.C_in*self.expansion, [16,24,32,48,64,96,112,144,184,192,336,352,384,552,672,1104])
        # e = convert_to_binary(self.C_out, [16,24,32,48,64,96,112,144,184,192,336,352,384,552,672,1104])
        # op_list.append([a,b,c,d,e])
        if self.C_in == self.C_out and self.stride == 1:
            x += identity

        return x,self.scale
    
    def get_arch(self):
        best_activ_list = []
        best_weight_list = []
        a,w = self.conv1.get_arch()
        best_activ_list.append(a)
        best_weight_list.append(w)
        a,w = self.conv2.get_arch()
        best_activ_list.append(a)
        best_weight_list.append(w)
        a,w = self.conv3.get_arch()
        best_activ_list.append(a)
        best_weight_list.append(w)
        return best_activ_list,best_weight_list



class Skip(nn.Module):
    def __init__(self, C_in, C_out, layer_id, stride=1,gpu = 0):
        super(Skip, self).__init__()
        assert stride in [1, 2]
        assert C_out % 2 == 0, 'C_out=%d'%C_out
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.scale = []
        self.layer_id = layer_id

        # self.register_buffer('active_bit_list', torch.tensor([0]).cuda())

        self.kernel_size = 1
        self.padding = 0

        if stride == 2 or C_in != C_out:
            self.conv = Conv2d(C_in, C_out, wbits = quant_bit , abits = quant_bit , share_weight = False , kernel_size = 1, stride=stride, padding=0, bias=False, gpu = gpu)
            self.bn = BatchNorm2d(C_out)
            self.relu = nn.ReLU(inplace=True)


    def forward(self, x,beta=None,quant_choose = 0):
        self.scale = []
        if hasattr(self, 'conv'):
            out,scale = self.conv(x, beta=beta,quant_choose = quant_choose)
            out = self.bn(out)
            out = self.relu(out)
            self.scale.append(scale)
        else:
            out = x

        return out,self.scale

    def get_arch(self):
        best_activ_list = []
        best_weight_list = []
        if hasattr(self, 'conv'):
            a,w = self.conv.get_arch()
            best_activ_list.append(a)
            best_weight_list.append(w)
        return best_activ_list,best_weight_list


class ConvNorm(nn.Module):
    '''
    conv => norm => activation
    use native Conv2d, not slimmable
    '''
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False,gpu = 0):
        super(ConvNorm, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.scale = []
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        if padding is None:
            # assume h_out = h_in / s
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.))
        else:
            self.padding = padding
        self.dilation = dilation
        assert type(groups) == int
        self.groups = groups
        self.bias = bias
        self.conv = Conv2d(C_in, C_out, wbits = [8] , abits = [8] , share_weight = False , kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, 
                            dilation=self.dilation, groups=self.groups, bias=bias,gpu = gpu)
        # self.conv = nn.Conv2d(C_in, C_out, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, 
        #                     dilation=self.dilation, groups=self.groups, bias=bias)
        self.bn = BatchNorm2d(C_out)

        self.relu = nn.ReLU(inplace=True)



    def forward(self, x, num_bits=32, beta=None, mode='soft', act_num=None, beta_param=None,quant_choose = 0):
        self.scale = []
        x,scale = self.conv(x,beta = beta,quant_choose = quant_choose)
        self.scale.append(scale)
        x = self.relu(self.bn(x))
        # op_list.append([self.kernel_size,self.stride,x.size()[2],self.C_in,self.C_out])
        # kernel [1,3,5] stride [1,2] size [32,16,8,4] c_in [16,24,32,48,64,96,112,144,184,192,336,352,384,552,672,1104] abit [2,4,8] wbit [2,4,8]
        # op_list.append([self.kernel_size,self.stride,x.size()[2],self.C_in,self.C_out])
        # print(convert_to_binary(self.kernel_size, [1, 3, 5]))
        # print(convert_to_binary(self.stride, [1, 2]))
        # print(convert_to_binary(x.size()[2], [32, 16, 8, 4]))
        # print(convert_to_binary(self.C_in, [16,24,32,48,64,96,112,144,184,192,336,352,384,552,672,1104]))
        # print(convert_to_binary(self.C_out, [16,24,32,48,64,96,112,144,184,192,336,352,384,552,672,1104]))
        return x,self.scale

def get_op_list():
    return op_list
def clean_op_list():
    op_list.clear()

OPS = {
    'k3_e1' : lambda C_in, C_out, layer_id, stride,gpu: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=1,gpu = gpu),
    'k3_e1_g2' : lambda C_in, C_out, layer_id, stride,gpu: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=2,gpu = gpu),
    'k3_e3' : lambda C_in, C_out, layer_id, stride,gpu: ConvBlock(C_in, C_out, layer_id, expansion=3, kernel_size=3, stride=stride, groups=1,gpu = gpu),
    'k3_e6' : lambda C_in, C_out, layer_id, stride,gpu: ConvBlock(C_in, C_out, layer_id, expansion=6, kernel_size=3, stride=stride, groups=1,gpu = gpu),
    'k5_e1' : lambda C_in, C_out, layer_id, stride,gpu: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=1,gpu = gpu),
    'k5_e1_g2' : lambda C_in, C_out, layer_id, stride,gpu: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=2,gpu = gpu),
    'k5_e3' : lambda C_in, C_out, layer_id, stride,gpu: ConvBlock(C_in, C_out, layer_id, expansion=3, kernel_size=5, stride=stride, groups=1,gpu = gpu),
    'k5_e6' : lambda C_in, C_out, layer_id, stride,gpu: ConvBlock(C_in, C_out, layer_id, expansion=6, kernel_size=5, stride=stride, groups=1,gpu = gpu),
    'skip' : lambda C_in, C_out, layer_id, stride,gpu: Skip(C_in, C_out, layer_id, stride,gpu = gpu)
}
# OPS = {
#     'k3_e1' : lambda C_in, C_out, layer_id, stride,gpu: ConvBlock_N(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=1),
#     'k3_e1_g2' : lambda C_in, C_out, layer_id, stride,gpu: ConvBlock_N(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=2),
#     'k3_e3' : lambda C_in, C_out, layer_id, stride,gpu: ConvBlock_N(C_in, C_out, layer_id, expansion=3, kernel_size=3, stride=stride, groups=1),
#     'k3_e6' : lambda C_in, C_out, layer_id, stride,gpu: ConvBlock_N(C_in, C_out, layer_id, expansion=6, kernel_size=3, stride=stride, groups=1),
#     'k5_e1' : lambda C_in, C_out, layer_id, stride,gpu: ConvBlock_N(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=1),
#     'k5_e1_g2' : lambda C_in, C_out, layer_id, stride,gpu: ConvBlock_N(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=2),
#     'k5_e3' : lambda C_in, C_out, layer_id, stride,gpu: ConvBlock_N(C_in, C_out, layer_id, expansion=3, kernel_size=5, stride=stride, groups=1),
#     'k5_e6' : lambda C_in, C_out, layer_id, stride,gpu: ConvBlock_N(C_in, C_out, layer_id, expansion=6, kernel_size=5, stride=stride, groups=1),
#     'skip' : lambda C_in, C_out, layer_id, stride,gpu: Skip(C_in, C_out, layer_id, stride)
# }
