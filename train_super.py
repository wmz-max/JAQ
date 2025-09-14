from __future__ import division
import os
import sys
import time
import glob
import logging
from tqdm import tqdm
import ast
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from torch.autograd import Variable
from operations_q import get_op_list
import torchvision

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

from tensorboardX import SummaryWriter

from config_train_super_q import config
# from accelerator_new.hardware import Accelerator
from datasets import prepare_train_data, prepare_test_data, prepare_train_data_for_search, prepare_test_data_for_search
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image

from model_super_q import FBNet_Super

from generator import Generator
from estimator import Area_estimator

from itertools import chain
best_acc = 0
best_epoch = 0
# from dance_modules.evaluator import Evaluator
NBA = False


def convert_to_binary(value, options):
    return [1 if value == opt else 0 for opt in options]


# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def main():

    ngpus_per_node = torch.cuda.device_count()
    if config.ddp :
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        main_worker(config.gpu,ngpus_per_node, config)


def main_worker(gpu, ngpus_per_node, config):
    global best_acc
    global best_epoch

    config.gpu = gpu
    pretrain = config.pretrain

    device = torch.device(f'cuda:{config.gpu}' if torch.cuda.is_available() else 'cpu')

    model = FBNet_Super(config=config)

    generator = Generator().to(device)
    generator_opt = torch.optim.Adam(generator.parameters(), lr=0.01)
    generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(generator_opt, 120 // 3, 0.0001)
    area_estimator = Area_estimator()
    area_estimator.load_state_dict(torch.load('/root/autodl-tmp/code_area/model_epoch_200.pth', map_location=device))
    area_estimator.to(device)
    area_estimator.eval()

    if config.ddp:
        config.rank = config.rank * ngpus_per_node + config.gpu
        dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                world_size=config.world_size, rank=config.rank)
        torch.cuda.set_device(config.gpu)
        model.cuda(config.gpu)

        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        config.batch_size = int(config.batch_size / ngpus_per_node)
        config.workers = int((config.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu], find_unused_parameters=True)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
    else:
        device = torch.device(f'cuda:{config.gpu}' if torch.cuda.is_available() else 'cpu')
        model.to(device)



    logger = SummaryWriter(config.save)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info("args = %s", str(config))


    parameters = []
    if hasattr(model, 'module'):
        parameters += list(model.module.stem.parameters())
        parameters += list(model.module.cells.parameters())
        parameters += list(model.module.header.parameters())
        parameters += list(model.module.fc.parameters())
    else:
        parameters += list(model.stem.parameters())
        parameters += list(model.cells.parameters())
        parameters += list(model.header.parameters())
        parameters += list(model.fc.parameters())


    if config.opt == 'Adam':
        optimizer = torch.optim.Adam(
            parameters,
            lr=config.lr,
            betas=config.betas)
    elif config.opt == 'Sgd':
        optimizer = torch.optim.SGD(
            parameters,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay)
    else:
        print("Wrong Optimizer Type.")
        sys.exit()

    if hasattr(model, 'module'):
        optimizer_arch = torch.optim.Adam(list(model.module._arch_params.values()), lr=3e-4, betas=(0.5, 0.999))
    else:
        optimizer_arch = torch.optim.Adam(list(model._arch_params.values()), lr=3e-4, betas=(0.5, 0.999))
    
    if config.lr_schedule == 'cosine':
        lr_policy = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(config.nepochs), eta_min=config.learning_rate_min)
    else:
        print("Wrong Learning Rate Schedule Type.")
        sys.exit()

    cudnn.benchmark = True

    start_epoch = 0
    print('No checkpoint. Train from scratch.')



    filename = '/root/autodl-tmp/code_area/op_list_encode_cifar100.txt'
    with open(filename, 'r') as file:
        lines = file.readlines()
    op_data = []
    for line in lines:
        if line.strip():
            current_list = ast.literal_eval(line.strip())
            op_data.append(current_list)


    # data loader ############################
    if 'cifar' in config.dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        if config.dataset == 'cifar10':
            train_data = dset.CIFAR10(root=config.dataset_path, train=True, download=True, transform=transform_train)
            test_data = dset.CIFAR10(root=config.dataset_path, train=False, download=True, transform=transform_test)
        elif config.dataset == 'cifar100':
            train_data = dset.CIFAR100(root=config.dataset_path, train=True, download=True, transform=transform_train)
            test_data = dset.CIFAR100(root=config.dataset_path, train=False, download=True, transform=transform_test)
            # indices = []
            # for i in range(100):  # 对于CIFAR100中的100个类
            #     start_idx = i * 500  # CIFAR100每个类有500个样本
            #     indices.extend(range(start_idx, start_idx + 50))  # 取前50个
            # train_data = torch.utils.data.Subset(train_data, indices)
        else:
            print('Wrong dataset.')
            sys.exit()

    if config.dataset == 'imagenet':
        train_data = prepare_train_data_for_search(dataset=config.dataset,
                                          datadir=config.dataset_path+'/train', num_class=config.num_classes)
        test_data = prepare_test_data_for_search(dataset=config.dataset,
                                        datadir=config.dataset_path+'/val', num_class=config.num_classes)


    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(config.train_portion * num_train))

    if config.ddp:
        train_data_model = torch.utils.data.Subset(train_data, indices[:split])
        train_data_arch = torch.utils.data.Subset(train_data, indices[split:num_train])
        train_sampler_model = torch.utils.data.distributed.DistributedSampler(train_data_model)
        train_sampler_arch = torch.utils.data.distributed.DistributedSampler(train_data_arch)
    else:
        train_sampler_model = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        train_sampler_arch = torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train])

    if config.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None

    train_loader_model = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size, 
        sampler=train_sampler_model, shuffle=(train_sampler_model is None),
        pin_memory=False, num_workers=config.num_workers, drop_last=True)
    train_loader_arch = torch.utils.data.DataLoader(
            train_data, batch_size=config.batch_size,
            sampler=train_sampler_arch, shuffle=(train_sampler_arch is None),
            pin_memory=False, num_workers=config.num_workers, drop_last=True)

    test_loader = torch.utils.data.DataLoader(test_data,
                                        batch_size=config.batch_size,
                                        shuffle=False,
                                        pin_memory=True,
                                        num_workers=config.num_workers)





    for epoch in range(start_epoch, config.nepochs):
        if config.ddp:
            train_sampler.set_epoch(epoch)

        scale = train(train_loader_model,train_loader_arch, model, optimizer, optimizer_arch,lr_policy, logger, epoch, config,op_data, generator, generator_opt, area_estimator)
        generator_scheduler.step()


        torch.cuda.empty_cache()
        lr_policy.step()

        
        eval_epoch = config.eval_epoch

        #validation
        if (epoch+1) % eval_epoch == 0:

            with torch.no_grad():
                acc = infer(epoch, model, test_loader, logger,config)

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch

            if not config.ddp or (config.ddp and config.rank % ngpus_per_node == 0):
                logging.info("Epoch:%d Acc:%.3f Best Acc:%.3f Best Epoch:%d" % (epoch, acc, best_acc, best_epoch))
                # save(model, os.path.join(config.save, 'weights_%d.pt'%epoch))

                state = {}
                state['state_dict'] = model.state_dict()
                if hasattr(model, 'module'):
                    state['alpha'] = getattr(model.module, 'alpha')
                    state['beta'] = getattr(model.module, 'beta')
                else:
                    state['alpha'] = getattr(model, 'alpha')
                    state['beta'] = getattr(model, 'beta')
                state['optimizer'] = optimizer.state_dict()
                state['optimizer_arch'] = optimizer_arch.state_dict()
                state['lr_scheduler'] = lr_policy.state_dict()
                state['epoch'] = epoch 

                torch.save(state, os.path.join(config.save, "arch_%d.pt"%(epoch)))





    del train_loader
    del test_loader
    torch.cuda.empty_cache()




def train(train_loader_model, train_loader_arch, model, optimizer, optimizer_arch,lr_policy, logger, epoch, config, op_data, generator, generator_opt, area_estimator):

    model.train()
    prec1_list = []
    prec1_arch_list = []
    device = torch.device(f'cuda:{config.gpu}' if torch.cuda.is_available() else 'cpu')

    dataloader_model = iter(train_loader_model)
    dataloader_arch = iter(train_loader_arch)

    for step in range(len(train_loader_model)):
        start_time = time.time()
        data_time = time.time() - start_time



        
        # op_list = get_op_list()
        # with open("op_list_encode_cifar100.txt", "w") as file:
        #     for item in op_list:
        #         file.write(str(item) + "\n")
        # sys.exit()

        # model_op_size_list = model.get_model_op_size_list()
        # with open("op_list_encode_cifar100.txt", "w") as file:
        #     for item in model_op_size_list:
        #         file.write(str(item) + "\n")
        # sys.exit()





        #############################arch############################
        try:
            input, target = next(dataloader_arch)
        except:
            dataloader_arch = iter(train_loader_arch)
            input, target = next(dataloader_arch)
        input = input.to(device , non_blocking=True)
        target = target.to(device , non_blocking=True)
        if hasattr(model, 'module'):
            criterion = model.module._criterion
        else:
            criterion = model._criterion
        if NBA is True:
            logit,scale = model(input,quant_choose = 2)
        else:
            logit,scale = model(input,quant_choose = 0)
        loss = criterion(logit, target)
        prec1, = accuracy(logit.data, target, topk=(1,))
        prec1_arch_list.append(prec1)
        loss.backward()


        alpha,beta =  model.module._get_arch_parameters()
        ###########old##################
        hw_input = []
        for i in range(22):
            op_index = F.softmax(alpha[i], dim=0).argmax(-1)
            op_0 = op_data[i][op_index][0][0] + op_data[i][op_index][0][1] + op_data[i][op_index][0][2] + op_data[i][op_index][0][3] + op_data[i][op_index][0][4]
            op_1 = op_data[i][op_index][1][0] + op_data[i][op_index][1][1] + op_data[i][op_index][1][2] + op_data[i][op_index][1][3] + op_data[i][op_index][1][4]
            op_2 = op_data[i][op_index][2][0] + op_data[i][op_index][2][1] + op_data[i][op_index][2][2] + op_data[i][op_index][2][3] + op_data[i][op_index][2][4]
            beta_activ_index = F.softmax(beta[i][op_index][0], dim=-1).argmax(-1)
            activ_q = [0,0,0]
            activ_q[beta_activ_index] = 1
            beta_weight_index = F.softmax(beta[i][op_index][1], dim=-1).argmax(-1)
            weight_q = [0,0,0]
            weight_q[beta_weight_index] = 1
            hw_input.append(op_0 + op_1 + op_2 + activ_q + weight_q) 
        
        hw_input_list = list(chain.from_iterable(hw_input))
        hw_input_tensor = torch.tensor(hw_input_list, dtype=torch.float).to(device)
        area = harware(hw_input_tensor,generator,generator_opt,area_estimator)
        

        hw_loss = area
        hw_loss_temp = hw_loss / 22
        hw_backward_loss = 0
        for i in range(22):
            ########op alpha
            op_index = F.softmax(alpha[i], dim=0).argmax(-1)
            alpha_use = alpha[i][op_index]
            alpha_ = (1-alpha_use).detach() + alpha_use
            ########beta_activ
            beta_activ_index = F.softmax(beta[i][op_index][0], dim=-1).argmax(-1)
            beta_activ_use = beta[i][op_index][0][beta_activ_index]
            beta_activ_ = (1-beta_activ_use).detach() + beta_activ_use
            ########beta_weight
            beta_weight_index = F.softmax(beta[i][op_index][1], dim=-1).argmax(-1)
            beta_weight_use = beta[i][op_index][0][beta_weight_index]
            beta_weight_ = (1-beta_weight_use).detach() + beta_weight_use
            hw_backward_loss += hw_loss_temp * alpha_ * beta_activ_ * beta_weight_
        
        # for i in range(22):
        #     ########op alpha
        #     op_softmax = F.softmax(alpha[i], dim=0)
        #     op_temp = 0
        #     for j in range(9):
        #         op_temp += hw_loss_temp * op_softmax[j]
        #     ########beta_activ
        #     # beta_activ_softmax = F.softmax(beta[i][op_index][0], dim=-1)
        #     # ########beta_weight
        #     # beta_weight_softmax = F.softmax(beta[i][op_index][1], dim=-1)
        #     # hw_backward_loss += hw_loss_temp * sum(op_softmax) * sum(beta_activ_softmax) * sum(beta_weight_softmax)
        #     hw_backward_loss += op_temp
        hw_backward_loss.backward()
        
        # print(alpha.grad)  # 检查alpha的梯度
        # print(beta.grad)   # 检查beta的梯度     
        
        #########################new#########################
        # hw_input = []
        # for i in range(22):
        #     op_index = F.softmax(alpha[i], dim=0).argmax(-1)
        #     op_softmax = F.softmax(alpha[i], dim=0)
        #     beta_activ_softmax = F.softmax(beta[i][op_index][0], dim=-1)
        #     beta_weight_softmax = F.softmax(beta[i][op_index][1], dim=-1)
        #     tensor = torch.cat([op_softmax,beta_activ_softmax,beta_weight_softmax], dim=0)
        #     hw_input.append(tensor)
        # final_tensor = torch.cat(hw_input, dim=0)
        # area = harware(final_tensor,generator,generator_opt,area_estimator)
        # hw_loss = area * 1e-7
        # hw_loss.backward()
        
        # print(alpha.grad)  # 检查alpha的梯度
        # print(beta.grad)   # 检查beta的梯度 
        
        optimizer_arch.step()
        optimizer_arch.zero_grad()
        
        
        ###########weight###############
        input, target = next(dataloader_model) 
        input = input.to(device , non_blocking=True)
        target = target.to(device , non_blocking=True)
        if hasattr(model, 'module'):
            criterion = model.module._criterion
        else:
            criterion = model._criterion
        
        
        if NBA is True:
            logit,scale = model(input,quant_choose = 1)
        else:
            logit,scale = model(input,quant_choose = 0)
        loss = criterion(logit, target)
        prec1, = accuracy(logit.data, target, topk=(1,))
        prec1_list.append(prec1)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        
        if step % 10 == 0:
            if not config.ddp or (config.ddp and config.rank % config.ngpus_per_node == 0):
                # logging.info(f"op_list: {op_list}")
                # logging.info(f"activ_list: {activ_list}")
                # logging.info(f"weight_list: {weight_list}")
                # logging.info("bitops = %.3f Gbitops", gbitops)
                total_time = time.time() - start_time
                logging.info("[Epoch %d/%d][Step %d/%d] Loss=%.3f Time=%.3f Data Time=%.3f" % (epoch + 1, config.nepochs, step + 1, len(train_loader_model), loss.item(), total_time, data_time))


    acc = sum(prec1_list)/len(prec1_list)
    acc_arch = sum(prec1_arch_list)/len(prec1_arch_list)
    torch.cuda.empty_cache()
    logging.info("train acc = %.3f",acc)
    logging.info("train arch acc = %.3f",acc_arch)
    del loss

    return scale



def harware(input,generator,generator_opt,area_estimator):
    generator.train()
    generator_opt.zero_grad()
    input_train = input.detach()
    hw_params = generator(input_train)
    area = area_estimator(hw_params)
    hw_loss = area
    hw_loss.backward()
    generator_opt.step()
    generator.eval()
    hw_params = generator(input)
    area = area_estimator(hw_params)
    return area
    



def infer(epoch, model, test_loader, logger,config):
    model.eval()
    prec1_list = []
    device = torch.device(f'cuda:{config.gpu}' if torch.cuda.is_available() else 'cpu')
    for i, (input, target) in enumerate(test_loader):
        input_var = input.cuda(device, non_blocking=True)
        target_var = target.cuda(device, non_blocking=True)
        output,_ = model(input_var)
        prec1, = accuracy(output.data, target_var, topk=(1,))
        prec1_list.append(prec1)

    acc = sum(prec1_list)/len(prec1_list)

    return acc



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save(model, model_path):
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    main()

