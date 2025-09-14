import torch
import os
import matplotlib.pyplot as plt

# 设置文件夹路径
folder_path = '/root/autodl-tmp/code_area/super_q/ours_auto-nba-interface_1_area'
output_folder = '/root/autodl-tmp/code_area/draw/ours_auto-nba-interface_1_area_active'
# 找到所有以'arch'开头并以'.pt'结尾的文件
files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith('arch') and f.endswith('.pt')]

# # 初始化beta值存储
betas = []

# 加载每个文件并提取beta
for file in files:
    data = torch.load(file)
    if 'beta' in data:
        betas.append(data['beta'])

# 绘制22个折线图
for i in range(22):
    for k in range(9):
        plt.figure(figsize=(10, 5))
        for j in range(3):
            # 生成每个beta值的折线图
            plt.plot(range(len(betas)), [beta[i,k][0][j].cpu().item() for beta in betas], label=f'Beta {j+1}')
        plt.title(f'Plot for Beta Index')
        plt.xlabel('File Index')
        plt.ylabel('Beta Value')
        plt.legend()
        plt.savefig(os.path.join(output_folder, f'beta_index_{i}_{k}.png'))
        plt.close()


# alphas = []

# # 加载每个文件并提取beta
# for file in files:
#     data = torch.load(file)
#     if 'alpha' in data:
#         alphas.append(data['alpha'])

# # 绘制22个折线图
# for i in range(22):
#     plt.figure(figsize=(10, 5))
#     for j in range(9):
#         # 生成每个beta值的折线图
#         plt.plot(range(len(alphas)), [alpha[i][j].cpu().item() for alpha in alphas], label=f'Alpha {j+1}')
#     plt.title(f'Plot for Alpha Index')
#     plt.xlabel('File Index')
#     plt.ylabel('Beta Value')
#     plt.legend()
#     plt.savefig(os.path.join(output_folder, f'alpha_index_{i}.png'))
#     plt.close()