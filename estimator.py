import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, random_split
from tensorboardX import SummaryWriter
import logging
import os
import sys

class Block(nn.Module):
    def __init__(self, num_features):
        super(Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
        )

    def forward(self, x):
        residual = x
        out = self.layer(x)
        out += residual
        return x

class Area_estimator(nn.Module):
    def __init__(self, num_mac_array_x = 62, num_mac_array_y = 62, num_weight_buffer = 29, num_activ_buffer = 29,num_output_buffer = 29):
        super(Area_estimator, self).__init__()
        len_hw_params = num_mac_array_x + num_mac_array_y + num_weight_buffer + num_activ_buffer + num_output_buffer
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Sequential(
                nn.Linear(len_hw_params, 2048),
                nn.ReLU()
            )
        )
        for i in range(3):
            self.layers.append(Block(2048))
        self.layers.append(
            nn.Linear(2048, 1),
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# class Latency_Power_estimator(nn.Module):
#     def __init__(self, op_params = 7, hw_params = (62 + 62 + 30 + 30 + 30 + 3), tile_params = (15 + 15 + 15 + 15 + 120)):
#         super(Latency_Power_estimator, self).__init__()
#         self.layers = nn.ModuleList()
#         self.layers.append(
#             nn.Sequential(
#                 nn.Linear((op_params + hw_params + tile_params), 256),
#                 nn.ReLU()
#             )
#         )
#         for i in range(3):
#             self.layers.append(Block(256))
#         self.layers.append(
#             nn.Linear(256, 2),
#         )

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x




# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         # correct_k = correct[:k].view(-1).float().sum(0)
#         correct_k = correct[:k].contiguous().view(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res



# inputs = []
# targets = []

# logger = SummaryWriter('./')
# log_format = '%(asctime)s %(message)s'
# logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
# fh = logging.FileHandler(os.path.join('./', 'log.txt'))
# fh.setFormatter(logging.Formatter(log_format))
# logging.getLogger().addHandler(fh)


# scale = 10
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# with open('/villa/mingzi/NIPS_2024/mingzi_copy/accelerator_new/area.txt', 'r') as file:
#     lines = file.readlines()  # 读取所有行
#     for i in range(0, len(lines), 2):  # 每次跳过两行
#         if i + 1 < len(lines):  # 确保有足够的行
#             input_line = lines[i].strip().split(',')
#             target_line = lines[i + 1].strip()
#             input_tensor = torch.tensor([float(x) for x in input_line], dtype=torch.float32)
#             target_tensor = torch.tensor([float(float(target_line) / scale)], dtype=torch.float32)
#             inputs.append(input_tensor)
#             targets.append(target_tensor)
# with open('/villa/mingzi/NIPS_2024/mingzi_copy/accelerator_new/area_1000000_1.txt', 'r') as file:
#     lines = file.readlines()  # 读取所有行
#     for i in range(0, len(lines), 2):  # 每次跳过两行
#         if i + 1 < len(lines):  # 确保有足够的行
#             input_line = lines[i].strip().split(',')
#             target_line = lines[i + 1].strip()
#             input_tensor = torch.tensor([float(x) for x in input_line], dtype=torch.float32)
#             target_tensor = torch.tensor([float(float(target_line) / scale)], dtype=torch.float32)
#             inputs.append(input_tensor)
#             targets.append(target_tensor)
# with open('/villa/mingzi/NIPS_2024/mingzi_copy/accelerator_new/area_1000000.txt', 'r') as file:
#     lines = file.readlines()  # 读取所有行
#     for i in range(0, len(lines), 2):  # 每次跳过两行
#         if i + 1 < len(lines):  # 确保有足够的行
#             input_line = lines[i].strip().split(',')
#             target_line = lines[i + 1].strip()
#             input_tensor = torch.tensor([float(x) for x in input_line], dtype=torch.float32)
#             target_tensor = torch.tensor([float(float(target_line) / scale)], dtype=torch.float32)
#             inputs.append(input_tensor)
#             targets.append(target_tensor)
# with open('/villa/mingzi/NIPS_2024/mingzi_copy/accelerator_new/area_data/area_500000_1.txt', 'r') as file:
#     lines = file.readlines()  # 读取所有行
#     for i in range(0, len(lines), 2):  # 每次跳过两行
#         if i + 1 < len(lines):  # 确保有足够的行
#             input_line = lines[i].strip().split(',')
#             target_line = lines[i + 1].strip()
#             input_tensor = torch.tensor([float(x) for x in input_line], dtype=torch.float32)
#             target_tensor = torch.tensor([float(float(target_line) / scale)], dtype=torch.float32)
#             inputs.append(input_tensor)
#             targets.append(target_tensor)
# with open('/villa/mingzi/NIPS_2024/mingzi_copy/accelerator_new/area_data/area_500000_2.txt', 'r') as file:
#     lines = file.readlines()  # 读取所有行
#     for i in range(0, len(lines), 2):  # 每次跳过两行
#         if i + 1 < len(lines):  # 确保有足够的行
#             input_line = lines[i].strip().split(',')
#             target_line = lines[i + 1].strip()
#             input_tensor = torch.tensor([float(x) for x in input_line], dtype=torch.float32)
#             target_tensor = torch.tensor([float(float(target_line) / scale)], dtype=torch.float32)
#             inputs.append(input_tensor)
#             targets.append(target_tensor)
# with open('/villa/mingzi/NIPS_2024/mingzi_copy/accelerator_new/area_data/area_500000.txt', 'r') as file:
#     lines = file.readlines()  # 读取所有行
#     for i in range(0, len(lines), 2):  # 每次跳过两行
#         if i + 1 < len(lines):  # 确保有足够的行
#             input_line = lines[i].strip().split(',')
#             target_line = lines[i + 1].strip()
#             input_tensor = torch.tensor([float(x) for x in input_line], dtype=torch.float32)
#             target_tensor = torch.tensor([float(float(target_line) / scale)], dtype=torch.float32)
#             inputs.append(input_tensor)
#             targets.append(target_tensor)
# with open('/villa/mingzi/NIPS_2024/mingzi_copy/accelerator_new/area_data/area_1000000_1.txt', 'r') as file:
#     lines = file.readlines()  # 读取所有行
#     for i in range(0, len(lines), 2):  # 每次跳过两行
#         if i + 1 < len(lines):  # 确保有足够的行
#             input_line = lines[i].strip().split(',')
#             target_line = lines[i + 1].strip()
#             input_tensor = torch.tensor([float(x) for x in input_line], dtype=torch.float32)
#             target_tensor = torch.tensor([float(float(target_line) / scale)], dtype=torch.float32)
#             inputs.append(input_tensor)
#             targets.append(target_tensor)
       
        
# inputs = torch.stack(inputs).to(device)
# targets = torch.stack(targets).to(device)
# dataset = TensorDataset(inputs, targets)
# train_size = int(0.95 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
# model = Area_estimator()
# model = model.to(device)
# optimizer = Adam(model.parameters(), lr=0.001)
# loss_func = nn.MSELoss()

# num_epochs = 1000
# for epoch in range(num_epochs):
#     model.train()
#     for batch_inputs, batch_targets in train_loader:
#         optimizer.zero_grad()
#         output = model(batch_inputs)
#         # logging.info(f"epoch {epoch}")
#         # logging.info(f"output value {output}")
#         # logging.info(f"targets value {batch_targets}")
#         loss = loss_func(output, batch_targets)
#         loss.backward()
#         optimizer.step()
#     logging.info(f"train loss {loss.item()}")
    
#     model.eval()
#     with torch.no_grad():
#         val_loss = 0
#         for batch_inputs, batch_targets in val_loader:
#             output = model(batch_inputs)
#             print(output[0])
#             val_loss += loss_func(output, batch_targets).item()
#         val_loss /= len(val_loader)
#     logging.info(f"Val loss {val_loss}")
    
#     # torch.save(model.state_dict(), f'./pth/model_epoch_{epoch+1}.pth')