"""
MADE 训练工具模块
================

本文件实现了MADE模型训练的工具函数，包括：
1. MAF (Masked Autoregressive Flow) 训练函数
2. MADE 训练函数
3. 损失计算和优化

这些函数提供了训练循环中的核心功能，支持CPU和GPU训练。

作者: RAPIER 开发团队
版本: 1.0
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import MultivariateNormal
from typing import List, Optional


def train_one_epoch_maf(model, epoch, optimizer, train_loader):
    """
    训练一个epoch的MAF模型
    
    该函数实现了MAF (Masked Autoregressive Flow) 模型的单轮训练。
    MAF是一种基于流的生成模型，通过自回归变换学习数据分布。
    
    参数:
        model: MAF模型实例
        epoch (int): 当前训练轮数
        optimizer: 优化器实例
        train_loader: 训练数据加载器
        
    返回:
        float: 平均训练损失
    """
    model.train()
    train_loss = 0
    
    for batch in train_loader:
        # 前向传播：获取变换后的变量u和对数行列式
        u, log_det = model.forward(batch.float())

        # 计算负对数似然损失
        # 对于标准正态分布，负对数似然 = 0.5 * u^2 + 常数 - log_det
        negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
        negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
        negloglik_loss -= log_det
        negloglik_loss = torch.mean(negloglik_loss)

        # 反向传播和参数更新
        negloglik_loss.backward()
        train_loss += negloglik_loss.item()
        optimizer.step()
        optimizer.zero_grad()

    # 计算平均损失
    avg_loss = np.sum(train_loss) / len(train_loader)
    print("Epoch: {} Average loss: {:.5f}".format(epoch, avg_loss))
    return avg_loss


def train_one_epoch_made(model, epoch, optimizer, train_loader, cuda_device: Optional[int] = None):
    """
    训练一个epoch的MADE模型
    
    该函数实现了MADE (Masked Autoencoder for Distribution Estimation) 模型的单轮训练。
    MADE通过掩码机制确保自回归性质，能够有效建模高维数据的条件分布。
    
    参数:
        model: MADE模型实例
        epoch (int): 当前训练轮数
        optimizer: 优化器实例
        train_loader: 训练数据加载器
        cuda_device (int, optional): CUDA设备ID，None表示使用CPU
        
    返回:
        float: 平均训练损失
    """
    # 设置CUDA设备
    if cuda_device != None:
        torch.cuda.set_device(cuda_device)
    
    model.train()
    train_loss = []
    
    for batch in train_loader:
        if cuda_device == None:
            # CPU训练路径
            out = model.forward(batch.float())
            # 将输出分割为均值和对数标准差
            mu, logp = torch.chunk(out, 2, dim=1)
            # 计算标准化残差
            u = (batch - mu) * torch.exp(0.5 * logp)

            # 计算负对数似然损失
            # 对于高斯分布：NLL = 0.5 * u^2 + 0.5 * log(2π) - 0.5 * log(σ^2)
            negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
            negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
            negloglik_loss -= 0.5 * torch.sum(logp, dim=1)

            negloglik_loss = torch.mean(negloglik_loss)
            train_loss.append(negloglik_loss)

            # 反向传播和参数更新
            negloglik_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        else:
            # GPU训练路径
            input = batch.float().cuda()
            out = model.forward(input)
            # 将输出分割为均值和对数标准差
            mu, logp = torch.chunk(out, 2, dim=1)
            # 计算标准化残差
            u = (input - mu) * torch.exp(0.5 * logp).cuda()

            # 计算负对数似然损失
            negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
            negloglik_loss += 0.5 * input.shape[1] * np.log(2 * math.pi)
            negloglik_loss -= 0.5 * torch.sum(logp, dim=1)

            negloglik_loss = torch.mean(negloglik_loss)
            # 将损失转移到CPU并转换为numpy数组
            train_loss.append(negloglik_loss.cpu().detach().numpy())

            # 反向传播和参数更新
            negloglik_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # 计算平均损失
    N = len(train_loader)
    avg_loss = np.sum(train_loss) / N

    print("Epoch: {} Average loss: {:.5f}".format(epoch, avg_loss))
    return avg_loss
