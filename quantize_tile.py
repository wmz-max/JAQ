from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import time
import random
import math
from torch.nn.parameter import Parameter
# from quantizer_chen_lsq.lsq import LsqQuan
# from quantizer_git_lsq.lsq import LsqQuan
from quantize_nba import UniformQuantize_STE
from quantize_nba import UniformQuantize
from quantize_nba import UniformQuantize_Channel
from thop import profile
from torch.nn import init
from config_train_super_q import config
# from accelerator.bitfusion.bitfusion.src.simulator.simulator import Simulator
config_file = 'bitfusion.ini'
verbose = True

LSQ = False
NBA_STE = False
NBA = True
WEIGHT_SHARE = False
CUT_TILE_ACT = False
CUT_TILE_WEIGHT = False
TILE = 3
ONE_PATH = False
CHHANNEL = False
CHANNEL_RANDON = True
PERCENT_TO_SELECT = 3

# device = torch.device(f'cuda:{config.gpu}' if torch.cuda.is_available() else 'cpu')

class MixQuantActiv(nn.Module):

    def __init__(self, in_channel,bits,stride,gpu = 0):
        super(MixQuantActiv, self).__init__()
        self.bits = bits #bits
        self.mix_activ = nn.ModuleList()
        self.stride = stride
        self.device = gpu
        if LSQ is True:
            for bit in self.bits:
                self.mix_activ.append(LsqQuan(in_channel,bit=bit,all_positive=True,symmetric=False,per_channel=True))
         

    def forward(self, input,beta_activ = None,quant_choose = 0):
        outs = []
        
        if quant_choose == 1:
            NBA_STE = True
            ONE_PATH = False
        elif quant_choose == 2:
            NBA_STE = False
            ONE_PATH = True
        else:
            NBA_STE = False
            ONE_PATH = False
        
        if NBA_STE is True:
            sw = F.softmax(beta_activ, dim=0)
            for i, bit in enumerate(self.bits):
                out,_ = UniformQuantize_STE.apply(input, bit)
                outs.append(out * sw[i])
            activ = sum(outs)
            return activ,0
        elif ONE_PATH is True:
            sw = F.softmax(beta_activ, dim=0)
            rank = sw.argsort(descending=True)
            out,scale = UniformQuantize.quantize(input,self.bits[rank[0]])
            activ = out * ((1-sw[rank[0]]).detach() + sw[rank[0]])
            return activ,0
        else:
            if NBA is True:
                if self.stride == 0:
                    sw = F.softmax(beta_activ, dim=0)
                    for i, bit in enumerate(self.bits):
                        out,scale = UniformQuantize.quantize(input, bit)
                        outs.append(out * sw[i])
                    activ = sum(outs)
                    return activ,scale
                else:
                    if CUT_TILE_ACT:
                        input_clone = input.clone()
                        sw = F.softmax(beta_activ, dim=0).to(self.device)
                        for i, bit in enumerate(self.bits):
                            out,scale = UniformQuantize.quantize(input[:, :, :(TILE * self.stride), :(TILE * self.stride)], bit)
                            outs.append(out * sw[i])
                        activ = sum(outs)
                        input_clone[:, :, :(TILE * self.stride), :(TILE * self.stride)] = activ
                        return input_clone,scale
                    elif CHHANNEL:
                        input_clone = input.clone()
                        sw = F.softmax(beta_activ, dim=0).to(self.device)
                        for i, bit in enumerate(self.bits):
                            out,scale = UniformQuantize.quantize(input[:, 0, :, :], bit)
                            outs.append(out * sw[i])
                        activ = sum(outs)
                        input_clone[:, 0, :, :] = activ
                        return input_clone,scale
                    elif CHANNEL_RANDON:
                        channels = input.shape[1] 
                        # selected_channel = random.randint(0, channels - 1)
                        num_to_select = math.ceil(channels * PERCENT_TO_SELECT / 100)
                        selected_channels = random.sample(range(channels), num_to_select)
                        selected_tensor = input[:, selected_channels, :, :]
                        input_clone = input.clone()
                        sw = F.softmax(beta_activ, dim=0).to(self.device)
                        for i, bit in enumerate(self.bits):
                            out,scale = UniformQuantize.quantize(selected_tensor, bit)
                            outs.append(out * sw[i])
                        activ = sum(outs)
                        input_clone[:, selected_channels , :, :] = activ
                        return input_clone,scale
                    else:
                        sw = F.softmax(beta_activ, dim=0)
                        for i, bit in enumerate(self.bits):
                            out,scale = UniformQuantize.quantize(input, bit)
                            outs.append(out * sw[i])
                        activ = sum(outs)
                        return activ,scale
                
        if LSQ is True:
            if self.stride == 0:
                sw = F.softmax(beta_activ, dim=0)
                for i, branch in enumerate(self.mix_activ):
                    outs.append(branch(input) * sw[i])
                activ = sum(outs)
                return activ
            else:
                if CUT_TILE_ACT:
                    input_clone = input.clone()
                    sw = F.softmax(beta_activ, dim=0).to(self.device)
                    for i, branch in enumerate(self.mix_activ):
                        outs.append(branch(input[:, :, :(TILE * self.stride), :(TILE * self.stride)]) * sw[i])
                    activ = sum(outs)
                    input_clone[:, :, :(TILE * self.stride), :(TILE * self.stride)] = activ
                    return input_clone
                elif CHHANNEL:
                    input_clone = input.clone()
                    sw = F.softmax(beta_activ, dim=0).to(self.device)
                    for i, branch in enumerate(self.mix_activ):
                        outs.append(branch(input[:, 0, :, :]) * sw[i])
                    activ = sum(outs)
                    input_clone[:, 0, :, :] = activ
                    return input_clone
                elif CHANNEL_RANDON:
                    channels = input.shape[1] 
                    # selected_channel = random.randint(0, channels - 1)
                    num_to_select = math.ceil(channels * PERCENT_TO_SELECT / 100)
                    selected_channels = random.sample(range(channels), num_to_select)
                    selected_tensor = input[:, selected_channels, :, :]
                    input_clone = input.clone()
                    sw = F.softmax(beta_activ, dim=0).to(self.device)
                    for i, branch in enumerate(self.mix_activ):
                        outs.append(branch(selected_tensor) * sw[i])
                    activ = sum(outs)
                    input_clone[:, selected_channels , :, :] = activ
                    return input_clone
                else:
                    sw = F.softmax(beta_activ, dim=0).to(self.device)
                    for i, branch in enumerate(self.mix_activ):
                        outs.append(branch(input) * sw[i])
                    activ = sum(outs)
                    return activ

class MixQuantConv2d(nn.Module):

    def __init__(self, inplane, outplane, bits,gpu = 0,**kwargs):
        super(MixQuantConv2d, self).__init__()
        assert not kwargs['bias']
        self.bits = bits #bits
        # self.alpha_weight = Parameter(torch.Tensor(len(self.bits)))
        # self.alpha_weight.data.fill_(0.01)
        self.conv_list = nn.ModuleList()
        self.steps = []
        self.device = gpu
        m = nn.Conv2d(inplane, outplane, **kwargs)
        self.mix_weight = nn.ModuleList()
        init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            init.constant_(m.bias, 0)
        
        if WEIGHT_SHARE:
            self.conv_list.append(m)
        else:
            for bit in self.bits:
                self.conv_list.append(m)
        
        if LSQ is True:
            for bit in self.bits:
                lsq = LsqQuan(inplane,bit=bit,all_positive=False,symmetric=True,per_channel=False)
                lsq.init_from(m.weight)
                self.mix_weight.append(lsq)


    def forward(self, input, beta_weight = None,quant_choose = 0):
        mix_quant_weight = []
        sw = F.softmax(beta_weight, dim=0).to(self.device)
        
        if quant_choose == 1:
            NBA_STE = True
            ONE_PATH = False
        elif quant_choose == 2:
            NBA_STE = False
            ONE_PATH = True
        else:
            NBA_STE = False
            ONE_PATH = False
            
        # if WEIGHT_SHARE:
        #     weight = self.conv_list[0].weight
        #     if CUT_TILE_WEIGHT:
        #         weight_temp = weight[:TILE, :, :, :]
        #         if LSQ:
        #             for i, bit in enumerate(self.bits):
        #                 mix_quant_weight.append(self.mix_weight[i](weight_temp) * sw[i])
        #         if NBA:
        #             for i, bit in enumerate(self.bits):
        #                 quant_weight,scale = UniformQuantize.quantize(weight_temp, bit)
        #                 mix_quant_weight.append(quant_weight * sw[i])
        #         quant_weight = sum(mix_quant_weight)
        #         final_weight = torch.cat((quant_weight, weight[TILE:, :, :, :]), dim=0)
        #     else:
        #         if LSQ:
        #             for i, bit in enumerate(self.bits):
        #                 mix_quant_weight.append(self.mix_weight[i](weight) * sw[i])
        #         if NBA:
        #             for i, bit in enumerate(self.bits):
        #                 quant_weight,scale = UniformQuantize.quantize(weight, bit)
        #                 mix_quant_weight.append(quant_weight * sw[i])
        #         quant_weight = sum(mix_quant_weight)
        #         final_weight = torch.cat((quant_weight, weight[TILE:, :, :, :]), dim=0)
            
            
        # else:
        if NBA_STE is True:
            for i, bit in enumerate(self.bits):
                weight = self.conv_list[i].weight
                quant_weight,__ = UniformQuantize_STE.apply(weight, bit)
                scaled_quant_weight = quant_weight * sw[i]
                mix_quant_weight.append(scaled_quant_weight)
            final_weight = sum(mix_quant_weight)
        elif ONE_PATH is True:
            rank = sw.argsort(descending=True)
            weight = self.conv_list[rank[0]].weight
            quant_weight,scale = UniformQuantize.quantize(weight, self.bits[rank[0]])
            final_weight = quant_weight * ((1-sw[rank[0]]).detach() + sw[rank[0]])
        else:
            if NBA is True:
                if CUT_TILE_WEIGHT is True:
                    for i, bit in enumerate(self.bits):
                        weight = self.conv_list[i].weight
                        weight_temp = weight[:TILE, :, :, :]
                        quant_weight,scale = UniformQuantize.quantize(weight_temp, bit)
                        mix_quant_weight.append(torch.cat((quant_weight, weight[TILE:, :, :, :]), dim=0) * sw[i])
                    final_weight = sum(mix_quant_weight)
                else:
                    for i, bit in enumerate(self.bits):
                        weight = self.conv_list[i].weight
                        quant_weight,scale = UniformQuantize.quantize(weight, bit)
                        mix_quant_weight.append(quant_weight * sw[i])
                    final_weight = sum(mix_quant_weight)
            if LSQ is True:
                if CUT_TILE_WEIGHT is True:
                    for i, bit in enumerate(self.bits):
                        weight = self.conv_list[i].weight
                        weight_temp = weight[:TILE, :, :, :]
                        quant_weight = self.mix_weight[i](weight_temp)
                        mix_quant_weight.append(torch.cat((quant_weight, weight[TILE:, :, :, :]), dim=0) * sw[i])
                    final_weight = sum(mix_quant_weight)
                else:
                    for i, bit in enumerate(self.bits):
                        weight = self.conv_list[i].weight
                        quant_weight = self.mix_weight[i](weight)
                        mix_quant_weight.append(quant_weight * sw[i])
                    final_weight = sum(mix_quant_weight)


            



        conv = self.conv_list[0]
        out = F.conv2d(
            input, final_weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)

        return out






class MixActivConv2d(nn.Module):

    def __init__(self, inplane, outplane, wbits=None, abits=None, share_weight=False,gpu = 0, **kwargs):
        super(MixActivConv2d, self).__init__()
        if wbits is None:
            self.wbits = [1, 2]
        else:
            self.wbits = wbits
        if abits is None:
            self.abits = [1, 2]
        else:
            self.abits = abits
        stride = kwargs['stride'] if 'stride' in kwargs else 1

        # build mix-precision branches
        self.mix_activ = MixQuantActiv(inplane,self.abits,stride,gpu = gpu)
        self.share_weight = share_weight
        # if share_weight:
        #     self.mix_weight = SharedMixQuantConv2d(inplane, outplane, self.wbits, **kwargs)
        # else:
        self.mix_weight = MixQuantConv2d(inplane, outplane, self.wbits,gpu = gpu,**kwargs)
        # complexities
        
        if isinstance(kwargs['kernel_size'], tuple):
            kernel_size = kwargs['kernel_size'][0] * kwargs['kernel_size'][1]
        else:
            kernel_size = kwargs['kernel_size'] * kwargs['kernel_size']
        self.param_size = inplane * outplane * kernel_size * 1e-6
        self.filter_size = self.param_size / float(stride ** 2.0)
        self.register_buffer('size_product', torch.tensor(0, dtype=torch.float))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.float))
        self.out_channel = outplane
        self.in_channel = inplane
        self.output_size = 0
        self.stride = stride
        self.kernel_size = kernel_size
        self.groups = kwargs.get('groups', 1)


    def forward(self, input,beta = None,quant_choose = 0):
        # print(quant_choose)
        # in_shape = input.shape
        # tmp = torch.tensor(in_shape[1] * in_shape[2] * in_shape[3] * 1e-3, dtype=torch.float)
        # self.memory_size.copy_(tmp)
        # tmp = torch.tensor(self.filter_size * in_shape[-1] * in_shape[-2], dtype=torch.float)
        # self.size_product.copy_(tmp)
        out,scale = self.mix_activ(input,beta[0],quant_choose = quant_choose)
        out = self.mix_weight(out,beta[1],quant_choose = quant_choose)
        # self.output_size = out.shape[3]
        return out,scale

    # def get_arch(self):
    #     arch = []
    #     prob_activ = F.softmax(self.mix_activ.alpha_activ, dim=0)
    #     prob_activ = prob_activ.detach().cpu().numpy()
    #     best_activ = prob_activ.argmax()
    #     prob_weight = F.softmax(self.mix_weight.alpha_weight, dim=0)
    #     prob_weight = prob_weight.detach().cpu().numpy()
    #     best_weight = prob_weight.argmax()

    #     return best_activ,best_weight

    
    # def get_bitops(self):
    #     prob_activ = F.softmax(self.mix_activ.alpha_activ, dim=0)
    #     prob_activ = prob_activ.detach().cpu().numpy()
    #     best_activ = prob_activ.argmax()
    #     prob_weight = F.softmax(self.mix_weight.alpha_weight, dim=0)
    #     prob_weight = prob_weight.detach().cpu().numpy()
    #     best_weight = prob_weight.argmax()
    #     abits = self.mix_activ.bits
    #     wbits = self.mix_weight.bits

    #     cin = self.in_channel
    #     kernel_ops = self.kernel_size
    #     output_elements = self.output_size * self.output_size * self.out_channel
    #     # cout x oW x oH
    #     total_ops = cin * output_elements * kernel_ops // self.groups
    #     total_bitops = total_ops * abits[best_activ] * wbits[best_weight]
    #     return total_bitops
    
    # def get_hardware_aware(self):
    #     return simulator.get_conv_cycles(self.kernel_size,self.output_size,self.stride,self.in_channel,self.out_channel,self.abits[0],self.wbits[0])



class SharedMixQuantLinear(nn.Module):

    def __init__(self, inplane, outplane, bits, **kwargs):
        super(SharedMixQuantLinear, self).__init__()
        assert not kwargs['bias']
        self.bits = bits
        self.alpha_weight = Parameter(torch.Tensor(len(self.bits)))
        self.alpha_weight.data.fill_(0.01)
        self.linear = nn.Linear(inplane, outplane, **kwargs)
        self.steps = []

        self.mix_weight = nn.ModuleList()
        
        m = nn.Linear(inplane, outplane, **kwargs)
        init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            init.constant_(m.bias, 0)
        
        if LSQ is True:
            for bit in self.bits:
                lsq = LsqQuan(inplane,bit=bit,all_positive=False,symmetric=True,per_channel=False)
                lsq.init_from(m.weight)
                self.mix_weight.append(lsq)


    def forward(self, input):
        mix_quant_weight = []
        sw = F.softmax(self.alpha_weight, dim=0)
        linear = self.linear
        weight = linear.weight

        for i, bit in enumerate(self.bits):
            if NBA is True:
                quant_weight,__ = UniformQuantize_STE.apply(weight, bit)
                scaled_quant_weight = quant_weight * sw[i]
                mix_quant_weight.append(scaled_quant_weight)

            if LSQ is True:
                mix_quant_weight.append(self.mix_weight[i](weight) * sw[i])
                  
        mix_quant_weight = sum(mix_quant_weight)


        out = F.linear(input, mix_quant_weight, linear.bias)
        return out


class MixActivLinear(nn.Module):

    def __init__(self, inplane, outplane, wbits=None, abits=None, share_weight=True, **kwargs):
        super(MixActivLinear, self).__init__()
        if wbits is None:
            self.wbits = [1, 2]
        else:
            self.wbits = wbits
        if abits is None:
            self.abits = [1, 2]
        else:
            self.abits = abits
        # build mix-precision branches
        self.mix_activ = MixQuantActiv(self.abits,0)
        assert share_weight
        self.share_weight = share_weight
        self.mix_weight = SharedMixQuantLinear(inplane, outplane, self.wbits, **kwargs)
        # complexities
        self.param_size = inplane * outplane * 1e-6
        self.register_buffer('size_product', torch.tensor(self.param_size, dtype=torch.float))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.float))
        self.out_channel = outplane
        self.in_channel = inplane


    def forward(self, input):
        tmp = torch.tensor(input.shape[1] * 1e-3, dtype=torch.float)
        self.memory_size.copy_(tmp)
        out = self.mix_activ(input)
        out = self.mix_weight(out)
        return out
    

    def get_bitops(self):
        prob_activ = F.softmax(self.mix_activ.alpha_activ, dim=0)
        prob_activ = prob_activ.detach().cpu().numpy()
        best_activ = prob_activ.argmax()
        prob_weight = F.softmax(self.mix_weight.alpha_weight, dim=0)
        prob_weight = prob_weight.detach().cpu().numpy()
        best_weight = prob_weight.argmax()
        abits = self.mix_activ.bits
        wbits = self.mix_weight.bits

        total_ops = self.in_channel * self.out_channel
        total_bitops = total_ops * abits[best_activ] * wbits[best_weight]
        return total_bitops

    def get_hardware_aware(self):
        return simulator.get_FC_cycles(self.in_channel,self.out_channel,8,8)
