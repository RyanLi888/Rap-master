"""
分轮次预测模块（MADE）
====================

从周期性保存的模型快照中读取模型，对指定测试集进行密度估计，
输出每个样本的负对数似然，用于后续的样本清理与阈值选择。

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

def predict_epochs(feat_dir, model_dir, made_dir, TRAIN, TEST, DEVICE, epoch):
    """
    使用指定epoch保存的模型对测试集进行预测
    
    参数:
        feat_dir (str): 特征目录
        model_dir (str): 模型目录（包含 epochs_* 模型）
        made_dir (str): 输出目录（保存负对数似然）
        TRAIN (str): 训练集标识
        TEST (str): 测试集标识
        DEVICE (str|int): CUDA设备ID
        epoch (int): 模型epoch标记
    """

    # --------- 参数设置 ----------
    model_name = 'made' # 'MAF' or 'MADE'
    dataset_name = 'myData'
    train_type = TRAIN
    test_type = TEST
    batch_size = 1024
    n_mades = 5
    hidden_dims = [512]
    lr = 1e-4
    random_order = False
    patience = 30  # 早停耐心
    seed = 290713
    cuda_device = int(DEVICE) if DEVICE != 'None' else None
    # -----------------------------------

    # 加载数据
    data = get_data(dataset_name, feat_dir, train_type, test_type)
    train = torch.from_numpy(data.train.x)
    # 数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(data, batch_size)
    # 输入维度
    n_in = data.n_dims
    # 模型文件名
    save_name = f"{model_name}_{dataset_name}_{train_type}_{'_'.join(str(d) for d in hidden_dims)}.pt"

    # 从快照加载模型
    model = torch.load(os.path.join(model_dir, 'epochs_' + save_name))

    if cuda_device != None:
        torch.cuda.set_device(cuda_device)
        model = model.cuda()

    # 测试得分（负对数似然）
    neglogP = test_made(model, test_loader, cuda_device)

    # 保存到文件
    with open(os.path.join(made_dir, '%s_%sMADE_%d'%(test_type, train_type, epoch)), 'w') as fp:
        for neglogp in neglogP:
            fp.write(str(float(neglogp)) + '\n')
