"""
MADE 分轮次训练与中间结果预测模块
=============================

本文件在训练过程中周期性保存模型并调用 predict_epochs 对中间结果进行推断，
用于后续的数据清理与阈值选择。

作者: RAPIER 开发团队
版本: 1.0
"""

import torch
import numpy as np
from .made import MADE
from .datasets.data_loaders import get_data, get_data_loaders
from .utils.train import train_one_epoch_made
from .utils.validation import val_made
import sys
import os
from .predict_epochs import predict_epochs
import re

# 训练MADE并记录训练过程中的loss
def main(feat_dir, model_dir, made_dir, TRAIN, DEVICE, MINLOSS):
    """
    分轮次训练MADE并在每10个epoch进行一次预测
    
    参数:
        feat_dir (str): 特征目录
        model_dir (str): 模型保存目录
        made_dir (str): 中间预测结果目录
        TRAIN (str): 训练数据类型
        DEVICE (str|int): CUDA设备ID
        MINLOSS (str|int): 最小训练损失阈值
    """

    # --------- 参数设置 ----------
    model_name = 'made'  # 'MAF' or 'MADE'
    dataset_name = 'myData'
    train_type = TRAIN
    batch_size = 128
    hidden_dims = [512]
    lr = 1e-4
    random_order = False
    patience = 50  # 早停耐心
    min_loss = int(MINLOSS)
    seed = 290713
    cuda_device = int(DEVICE) if DEVICE != 'None' else None
    plot = True
    max_epochs = 2000
    # -----------------------------------

    # 清理中间结果目录
    for filename in os.listdir(made_dir):
        os.system('rm ' + os.path.join(made_dir, filename))
            
    # 加载数据集
    data = get_data(dataset_name, feat_dir, train_type, train_type)
    train = torch.from_numpy(data.train.x)
    # 数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(data, batch_size)
    # 初始化模型
    n_in = data.n_dims
    model = MADE(n_in, hidden_dims, random_order=random_order, seed=seed, gaussian=True, cuda_device=cuda_device)

    # 优化器
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    if cuda_device != None:
        torch.cuda.set_device(cuda_device)
        model = model.cuda()

    # 模型文件名
    save_name = f"{model_name}_{dataset_name}_{train_type}_{'_'.join(str(d) for d in hidden_dims)}.pt"
    # 训练历史
    epochs_list = []
    train_losses = []
    val_losses = []
    # 早停
    i = 0
    max_loss = np.inf
    
    # 训练循环
    for epoch in range(1, max_epochs):
        train_loss = train_one_epoch_made(model, epoch, optimiser, train_loader, cuda_device)
        val_loss = val_made(model, val_loader, cuda_device)

        epochs_list.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 每10个epoch保存一次并进行中间预测
        if epoch % 10 == 0:
            model = model.cpu()
            torch.save(
                model, os.path.join(model_dir, 'epochs_' + save_name)
            )  # 第一个epoch会有UserWarning
            if cuda_device != None:
                model = model.cuda()

            predict_epochs(feat_dir, model_dir, made_dir, TRAIN, 'be', DEVICE, epoch)
            predict_epochs(feat_dir, model_dir, made_dir, TRAIN, 'ma', DEVICE, epoch)

        # 早停：保存验证集上最优的模型
        if val_loss < max_loss and train_loss > min_loss:
            i = 0
            max_loss = val_loss
            model = model.cpu()
            torch.save(
                model, os.path.join(model_dir, save_name)
            )  # 第一个epoch会有UserWarning
            if cuda_device != None:
                model = model.cuda()
            
        else:
            i += 1

        if i < patience:
            print("Patience counter: {}/{}".format(i, patience))
        else:
            print("Patience counter: {}/{}\n Terminate training!".format(i, patience))
            break

