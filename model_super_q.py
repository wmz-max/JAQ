import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from operations_q import *
from operations_q import get_op_list,clean_op_list
# from operations_nba import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
import numpy as np
from thop import profile
from matplotlib import pyplot as plt
from quantize_tile import MixActivConv2d,MixActivLinear
import time
from config_train_super_q import config

# device = torch.device(f'cuda:{config.gpu}' if torch.cuda.is_available() else 'cpu')

layer_op_size_list = []
model_op_size_list = []

class MixedOp(nn.Module):

    def __init__(self, C_in, C_out, layer_id, stride=1,gpu = 0):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.scale = []
        for primitive in PRIMITIVES:
            op = OPS[primitive](C_in, C_out, layer_id, stride,gpu)
            self._ops.append(op)




    def forward(self, x, alpha,beta,quant_choose = 0):
        self.scale = []
        result = 0
        rank = alpha.argsort(descending=True)

        # for i in range(9):
        #     x_temp,scale = self._ops[rank[i]](x,beta[rank[i]])
        #     result = result + x_temp * alpha[rank[i]]
        #     self.scale.append(scale)
        #     # result = result + self._ops[rank[i]](x,beta[rank[i]]) * alpha[rank[i]]

        for i in range(9):
            # clean_op_list()
            x_temp,scale = self._ops[i](x,beta[i],quant_choose)
            result = result + x_temp * alpha[i]
            self.scale.append(scale)
            # layer_op_size_list.append(get_op_list()[:])
            # clean_op_list()
        # model_op_size_list.append(layer_op_size_list[:])
        # layer_op_size_list.clear()
        




        return result,self.scale




class FBNet_Super(nn.Module):
    def __init__(self, config):
        super(FBNet_Super, self).__init__()

        self.num_classes = config.num_classes

        self.num_layer_list = config.num_layer_list
        self.num_channel_list = config.num_channel_list
        self.stride_list = config.stride_list

        self.num_bits_list = config.num_bits_list

        self.stem_channel = config.stem_channel
        self.header_channel = config.header_channel

        if config.dataset == 'imagenet':
            stride_init = 2
        else:
            stride_init = 1
        self.gpu = config.gpu
        self.stem = ConvNorm(3, self.stem_channel, kernel_size=3, stride=stride_init, padding=1, bias=False,gpu = self.gpu)

        self.cells = nn.ModuleList()

        layer_id = 1
        for stage_id, num_layer in enumerate(self.num_layer_list):
            for i in range(num_layer):
                if i == 0:
                    if stage_id == 0:
                        op = MixedOp(self.stem_channel, self.num_channel_list[stage_id], layer_id, stride=self.stride_list[stage_id],gpu = self.gpu)
                    else:
                        op = MixedOp(self.num_channel_list[stage_id-1], self.num_channel_list[stage_id], layer_id, stride=self.stride_list[stage_id],gpu = self.gpu)
                else:
                    op = MixedOp(self.num_channel_list[stage_id], self.num_channel_list[stage_id], layer_id, stride=1,gpu = self.gpu)
                
                layer_id += 1
                self.cells.append(op)

        self.header = ConvNorm(self.num_channel_list[-1], self.header_channel, kernel_size=1,gpu = self.gpu)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.header_channel, self.num_classes)
        # self.fc = MixActivLinear(self.header_channel, self.num_classes,wbits=[8], abits=[8], share_weight=True, bias = False)
        
        self._arch_params = self._build_arch_parameters()
        # self._reset_arch_parameters()

        self._criterion = nn.CrossEntropyLoss()
        self.scale = []

        



    def forward(self, input, temp=8, quant_choose = 0):
        self.scale = []
        alpha = nn.functional.gumbel_softmax(getattr(self, "alpha").to(self.gpu),temp)
        beta = nn.functional.gumbel_softmax(getattr(self, "beta").to(self.gpu),temp)
        # alpha = getattr(self, "alpha").to(device)
        # beta = getattr(self, "beta").to(device)
        out,scale = self.stem(input,beta = torch.tensor([[1.],[1.]]), quant_choose = quant_choose)
        self.scale.append(scale)
        for i, cell in enumerate(self.cells):
            out,scale = cell(out, alpha[i], beta[i],quant_choose = quant_choose)
            self.scale.append(scale)
        x,scale = self.header(out,beta = torch.tensor([[1.],[1.]]),quant_choose = quant_choose)
        out = self.fc(self.avgpool(x).view(out.size(0), -1))
        self.scale.append(scale)

        return out,self.scale


    def get_arch(self):
        activ_list = []
        weight_list = []
        op_data = getattr(self, "alpha").data
        op_list = F.softmax(op_data, dim=-1).argmax(-1)
        for i in range(22):
            op = self.cells[i]._ops[op_list[i]]
            a,w = op.get_arch()
            activ_list.append(a)
            weight_list.append(w)
        return op_list,activ_list,weight_list

    def fetch_best_arch(self):
        sum_bitops, sum_bita, sum_bitw = 0, 0, 0
        sum_mixbitops, sum_mixbita, sum_mixbitw = 0, 0, 0
        layer_idx = 0
        best_arch = None
        for m in self.modules():
            # print(m)
            if isinstance(m, MixActivConv2d) or isinstance(m, MixActivLinear):
                # print("hhhh")
                layer_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw = m.fetch_best_arch(layer_idx)
                if best_arch is None:
                    best_arch = layer_arch
                else:
                    for key in layer_arch.keys():
                        if key not in best_arch:
                            best_arch[key] = layer_arch[key]
                        else:
                            best_arch[key].append(layer_arch[key][0])
                sum_bitops += bitops
                sum_bita += bita
                sum_bitw += bitw
                sum_mixbitops += mixbitops
                sum_mixbita += mixbita
                sum_mixbitw += mixbitw
                layer_idx += 1
        return best_arch, sum_bitops, sum_bita, sum_bitw, sum_mixbitops, sum_mixbita, sum_mixbitw

    def _build_arch_parameters(self):
        num_ops = len(PRIMITIVES)
        device = torch.device(f'cuda:{self.gpu}' if torch.cuda.is_available() else 'cpu')
        setattr(self, 'alpha', nn.Parameter(Variable(1e-3*torch.ones(sum(self.num_layer_list), num_ops).to(device), requires_grad=True)))
        setattr(self, 'beta', nn.Parameter(Variable(1e-3*torch.ones(sum(self.num_layer_list), num_ops, 2, len(self.num_bits_list)).to(device), requires_grad=True)))
        return {"alpha": self.alpha, "beta": self.beta}


    def _reset_arch_parameters(self):
        num_ops = len(PRIMITIVES)
        device = torch.device(f'cuda:{self.gpu}' if torch.cuda.is_available() else 'cpu')
        getattr(self, "alpha").data = Variable(1e-3*torch.ones(sum(self.num_layer_list), num_ops).to(device), requires_grad=True)


    def _get_arch_parameters(self):
        return getattr(self, "alpha"), getattr(self, "beta")

    def find_current_single(self):
        op_list = [0] * (9 * 22)
        current_single = [0] * (15 * 22)
        alpha_save = []
        beta_activ_save = []
        beta_weight_save = []
        op_data = getattr(self, "alpha").data
        for i in range(22):
            op_index = F.softmax(op_data[i], dim=-1).argmax(-1)
            alpha_save.append(op_data[i][op_index])
            beta_activ = getattr(self, "beta").data[i][op_index][0]
            beta_activ_index = F.softmax(beta_activ, dim=-1).argmax(-1)
            beta_activ_save.append(beta_activ[beta_activ_index])
            beta_weight = getattr(self, "beta").data[i][op_index][1]
            beta_weight_index = F.softmax(beta_weight, dim=-1).argmax(-1)
            beta_weight_save.append(beta_weight[beta_weight_index])
            pos_op_0 = i * 9 + op_index
            pos_op = i * 15 + op_index
            pos_activ = i * 15 + 9 + beta_activ_index
            pos_weight = i * 15 + 12 + beta_weight_index
            op_list[pos_op_0] = 1
            current_single[pos_op] = 1
            current_single[pos_activ] = 1
            current_single[pos_weight] = 1
        return current_single,op_list,alpha_save,beta_activ_save,beta_weight_save
    
    def find_dance_current_single(self):
        op_list = [0] * (14 * 9)
        current_single = [0] * (15 * 22)
        alpha_save = []
        beta_activ_save = []
        beta_weight_save = []
        op_data = getattr(self, "alpha").data
        for i in range(14):
            op_index = F.softmax(op_data[i], dim=-1).argmax(-1)
            alpha_save.append(op_data[i][op_index])
            beta_activ = getattr(self, "beta").data[i][op_index][0]
            beta_activ_index = F.softmax(beta_activ, dim=-1).argmax(-1)
            beta_activ_save.append(beta_activ[beta_activ_index])
            beta_weight = getattr(self, "beta").data[i][op_index][1]
            beta_weight_index = F.softmax(beta_weight, dim=-1).argmax(-1)
            beta_weight_save.append(beta_weight[beta_weight_index])
            pos_op_0 = i * 7 + op_index
            pos_op = i * 15 + op_index
            pos_activ = i * 15 + 9 + beta_activ_index
            pos_weight = i * 15 + 12 + beta_weight_index
            op_list[pos_op_0] = 1
            current_single[pos_op] = 1
            current_single[pos_activ] = 1
            current_single[pos_weight] = 1
        return op_list
    
    def get_model_op_size_list(self):
        return model_op_size_list