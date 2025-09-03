"""
MADE 测试工具模块
================

本文件提供了MAF和MADE模型的测试函数，用于计算测试集上的负对数似然损失。

主要功能：
1. test_maf: MAF模型的测试损失计算
2. test_made: MADE模型的测试损失计算（返回每个样本的分数）

作者: RAPIER 开发团队
版本: 1.0
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import MultivariateNormal
from typing import Optional

def test_maf(model, train, test_loader):
    """
    计算MAF模型在测试集上的损失
    
    参数:
        model: MAF模型实例
        train: 训练数据（用于预热模型）
        test_loader: 测试数据加载器
        
    返回:
        None: 仅打印测试损失统计信息
    """
    model.eval()
    test_loss = []
    # 预热模型
    _, _ = model.forward(train)
    
    with torch.no_grad():
        for batch in test_loader:
            u, log_det = model.forward(batch.float())

            # 计算负对数似然损失
            negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
            negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
            negloglik_loss -= log_det

            test_loss.extend(negloglik_loss)
    N = len(test_loss)
    print(
        "Test loss: {:.4f} +/- {:.4f}".format(
            np.mean(test_loss), 2 + np.std(test_loss) / np.sqrt(N)
        )
    )


def test_made(model, test_loader, cuda_device: Optional[int]=None):
    """
    计算MADE模型在测试集上的负对数似然分数
    
    参数:
        model: MADE模型实例
        test_loader: 测试数据加载器
        cuda_device: CUDA设备ID，None表示使用CPU
        
    返回:
        list: 每个测试样本的负对数似然分数列表
    """
    if cuda_device != None:
        torch.cuda.set_device(cuda_device)
    model.eval()
    neglogP = []
    
    with torch.no_grad():
        for batch in test_loader:
            if cuda_device == None:
                # CPU路径
                input = batch.float()
                out = model.forward(input)
                mu, logp = torch.chunk(out, 2, dim=1)
                u = (input - mu) * torch.exp(0.5 * logp)

                # 计算负对数似然损失
                negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
                negloglik_loss += 0.5 * input.shape[1] * np.log(2 * math.pi)
                negloglik_loss -= 0.5 * torch.sum(logp, dim=1)

                neglogP.extend(negloglik_loss)
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
                neglogP.extend(negloglik_loss.cpu())
    
    print(len(neglogP))
    return neglogP


