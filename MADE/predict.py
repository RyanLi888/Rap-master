"""
使用MADE进行密度估计并导出分数
=============================

本文件载入训练完成的MADE模型，计算指定测试集每个样本的负对数似然，
结果写入文件供后续清理与阈值选择使用。

作者: RAPIER 开发团队
版本: 1.0
"""

import torch
import os
import numpy as np
from .made import MADE
from .datasets.data_loaders import get_data, get_data_loaders
from .utils.test import test_made
import sys
import os

# 使用MADE计算每个样本的密度估计
def main(feat_dir, model_dir, made_dir, TRAIN, TEST, DEVICE):
    """
    计算负对数似然并保存
    
    参数:
        feat_dir (str): 特征目录
        model_dir (str): 模型目录
        made_dir (str): 输出目录
        TRAIN (str): 训练数据类型
        TEST (str): 测试数据类型
        DEVICE (str|int): CUDA设备ID
    """

    # --------- 参数设置 ----------
    model_name = 'made' # 'MAF' or 'MADE'
    dataset_name = 'myData'
    train_type = TRAIN
    test_type = TEST
    batch_size = 1024
    hidden_dims = [512]
    cuda_device = int(DEVICE) if DEVICE != 'None' else None
    # -----------------------------------

    # 加载数据
    data = get_data(dataset_name, feat_dir, train_type, test_type)
    train = torch.from_numpy(data.train.x)
    # 数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(data, batch_size)
    # 模型文件名
    save_name = f"{model_name}_{dataset_name}_{train_type}_{'_'.join(str(d) for d in hidden_dims)}.pt"

    # 加载模型
    model = torch.load(os.path.join(model_dir, save_name))

    if cuda_device != None:
        torch.cuda.set_device(cuda_device)
        model = model.cuda()

    # 计算负对数似然
    neglogP = test_made(model, test_loader, cuda_device)

    # 保存结果
    with open(os.path.join(made_dir, '%s_%sMADE'%(test_type, train_type)), 'w') as fp:
        for neglogp in neglogP:
            fp.write(str(float(neglogp)) + '\n')
