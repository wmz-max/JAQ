import math
import torch
import torch.nn as nn
from torch.nn.functional import gumbel_softmax, softmax

from cosine_temperature import CosineTemperature


class Generator(nn.Module):
    # def __init__(self, num_blocks=22, num_ops=9, activ_num_quant = 3,weight_num_quant = 3, num_mac_array_x = 62, num_mac_array_y = 62, num_weight_buffer = 30, num_activ_buffer = 30,num_output_buffer = 30, num_df = 3, use_gumbel=True, gumbel_tau=5,
    #             total_epoch=120):
    def __init__(self, num_blocks=22, num_ops=129, num_mac_array_x = 62, num_mac_array_y = 62, num_weight_buffer = 29, num_activ_buffer = 29,num_output_buffer = 29, num_df = 3, use_gumbel=True, gumbel_tau=5,
                total_epoch=120):
        super(Generator, self).__init__()
        self.use_gumbel = use_gumbel
        self.temperature = CosineTemperature(eta_max=gumbel_tau, eta_min=0.5, total_epoch=total_epoch)
        self.temperature.update_tau(0)

        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Sequential(
                nn.Linear(num_blocks * num_ops, 128, bias=False),#15 * 22
                # nn.BatchNorm1d(128),
                nn.Sigmoid(),
            )
        )
        for i in range(3):
            self.layers.append(
                Block(128)
            )

        self.fc_array_x = nn.Sequential(
            nn.Linear(128, num_mac_array_x, bias=False),
            # nn.BatchNorm1d(num_mac_array_x),
        )
        self.fc_array_y = nn.Sequential(
            nn.Linear(128, num_mac_array_y, bias=False),
            # nn.BatchNorm1d(num_mac_array_y),
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(128, num_weight_buffer, bias=False),
            # nn.BatchNorm1d(num_weight_buffer),
        )
        self.fc_activ = nn.Sequential(
            nn.Linear(128, num_activ_buffer, bias=False),
            # nn.BatchNorm1d(num_activ_buffer),
        )
        self.fc_output = nn.Sequential(
            nn.Linear(128, num_output_buffer, bias=False),
            # nn.BatchNorm1d(num_output_buffer),
        )
        self.fc_df = nn.Sequential(
            nn.Linear(128, num_df, bias=False),
            # nn.BatchNorm1d(num_df),
        )


    def forward(self, x, eval_gumbel=True):
        for layer in self.layers:
            x = layer(x)

        array_x = self.fc_array_x(x)
        array_y = self.fc_array_y(x)
        weight = self.fc_weight(x)
        activ = self.fc_activ(x)
        output = self.fc_output(x)
        df = self.fc_df(x)
        if self.use_gumbel:
            array_x = gumbel_softmax(array_x, tau=self.temperature.tau, hard=False)
            array_y = gumbel_softmax(array_y, tau=self.temperature.tau, hard=False)
            weight = gumbel_softmax(weight, tau=self.temperature.tau, hard=False)
            activ = gumbel_softmax(activ, tau=self.temperature.tau, hard=False)
            output = gumbel_softmax(output, tau=self.temperature.tau, hard=False)
            df = gumbel_softmax(df, tau=self.temperature.tau, hard=False)

        #return torch.cat([array_x,array_y , weight, activ, output,df], dim=-1)
        return torch.cat([array_x,array_y , weight, activ, output], dim=-1)
            

class Block(nn.Module):
    def __init__(self, num_features):
        super(Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(num_features, num_features, bias=False),
            # nn.BatchNorm1d(num_features),
        )

    def forward(self, x):
        residual = x
        x = self.layer(x)
        x = torch.sigmoid(x).clone()
        x += residual
        return x

