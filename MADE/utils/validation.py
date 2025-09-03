"""
MADE 验证工具模块
================

本文件提供了MAF和MADE模型的验证函数，用于计算验证集上的负对数似然损失。

主要功能：
1. val_maf: MAF模型的验证损失计算
2. val_made: MADE模型的验证损失计算

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


def val_maf(model, train, val_loader):
    """
    计算MAF模型在验证集上的损失
    
    参数:
        model: MAF模型实例
        train: 训练数据（用于预热模型）
        val_loader: 验证数据加载器
        
    返回:
        float: 平均验证损失
    """
    model.eval()
    val_loss = []
    # 预热模型（避免第一次前向传播的延迟）
    _, _ = model.forward(train.float())
    
    for batch in val_loader:
        u, log_det = model.forward(batch.float())
        # 计算负对数似然损失
        negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
        negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
        negloglik_loss -= log_det
        val_loss.extend(negloglik_loss.tolist())

    N = len(val_loader.dataset)
    loss = np.sum(val_loss) / N
    print(
        "Validation loss: {:.4f} +/- {:.4f}".format(
            loss, 2 * np.std(val_loss) / np.sqrt(N)
        )
    )
    return loss


def val_made(model, val_loader, cuda_device: Optional[int]=None):
    """
    计算MADE模型在验证集上的损失
    
    参数:
        model: MADE模型实例
        val_loader: 验证数据加载器
        cuda_device: CUDA设备ID，None表示使用CPU
        
    返回:
        float: 平均验证损失
    """
    if cuda_device != None:
        torch.cuda.set_device(cuda_device)
    model.eval()
    val_loss = []
    
    with torch.no_grad():
        for batch in val_loader:
            if cuda_device == None:
                # CPU路径
                out = model.forward(batch.float())
                mu, logp = torch.chunk(out, 2, dim=1)
                u = (batch - mu) * torch.exp(0.5 * logp)

                # 计算负对数似然损失
                negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
                negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
                negloglik_loss -= 0.5 * torch.sum(logp, dim=1)
                negloglik_loss = torch.mean(negloglik_loss)

                val_loss.append(negloglik_loss.cpu())
            else:
                # GPU路径
                input = batch.float().cuda()
                out = model.forward(input)
                mu, logp = torch.chunk(out, 2, dim=1)
                u = (input - mu) * torch.exp(0.5 * logp).cuda()

                # 计算负对数似然损失
                negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
                negloglik_loss += 0.5 * input.shape[1] * np.log(2 * math.pi)
                negloglik_loss -= 0.5 * torch.sum(logp, dim=1)
                negloglik_loss = torch.mean(negloglik_loss)

                val_loss.append(negloglik_loss.cpu())

    N = len(val_loader)
    loss = np.sum(val_loss) / N
    print(
        "Validation loss: {:.4f} +/- {:.4f}".format(
            loss, 2 * np.std(val_loss) / np.sqrt(N)
        )
    )
    return loss
