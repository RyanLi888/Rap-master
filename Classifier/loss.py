"""
Co-teaching 损失函数
===================

本文件实现了Co-teaching训练策略中的损失函数：
两个模型在每个批次中分别选择对方的小损失样本进行更新，
以减少噪声标签对训练的影响。

作者: RAPIER 开发团队
版本: 1.0
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# 损失函数 - Co-teaching

def loss_coteaching(y_1, y_2, t, forget_rate):
    """
    Co-teaching损失计算
    
    参数:
        y_1 (Tensor): 模型1输出的logits (N, C)
        y_2 (Tensor): 模型2输出的logits (N, C)
        t (Tensor): 真实标签 (N,)
        forget_rate (float): 忘记率（丢弃大损失样本的比例）
    返回:
        tuple(Tensor, Tensor): 模型1和模型2各自的平均损失
    """
    # 计算各自的交叉熵损失（未归约）
    loss_1 = F.cross_entropy(y_1, t, reduce = False)
    ind_1_sorted = np.argsort(loss_1.data.cpu())
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduce = False)
    ind_2_sorted = np.argsort(loss_2.data.cpu())
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]
    # 交叉选择对方的小损失样本
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember

