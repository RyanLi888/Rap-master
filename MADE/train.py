"""
MADE 模型训练模块
================

本文件实现了MADE模型的训练流程，包括：
1. 模型参数配置
2. 数据加载和预处理
3. 训练循环和早停机制
4. 模型保存和验证

作者: RAPIER 开发团队
版本: 1.0
"""

import torch
import numpy as np
from .made import MADE
from .datasets.data_loaders import get_data, get_data_loaders
from .utils.train import train_one_epoch_made
from .utils.validation import val_made
import os

def main(feat_dir, model_dir, TRAIN, DEVICE, MINLOSS):
    """
    MADE模型训练主函数
    
    该函数实现了完整的MADE模型训练流程，包括数据加载、模型训练、
    验证和早停机制。
    
    参数:
        feat_dir (str): 特征数据目录路径
        model_dir (str): 模型保存目录路径
        TRAIN (str): 训练数据类型标识
        DEVICE (str): CUDA设备ID，'None'表示使用CPU
        MINLOSS (str): 最小损失阈值，用于早停判断
    """
    
    print("开始MADE模型训练...")
    
    # --------- 设置训练参数 ----------
    model_name = 'made'                    # 模型名称
    dataset_name = 'myData'                # 数据集名称
    train_type = TRAIN                     # 训练数据类型
    test_type = TRAIN                      # 测试数据类型（与训练数据相同）
    batch_size = 128                       # 批次大小
    hidden_dims = [512]                    # 隐藏层维度列表
    lr = 1e-4                             # 学习率
    random_order = False                   # 是否使用随机输入排序
    patience = 50                          # 早停耐心值
    min_loss = int(MINLOSS)               # 最小损失阈值
    seed = 290713                         # 随机种子
    cuda_device = int(DEVICE) if DEVICE != 'None' else None  # CUDA设备ID
    max_epochs = 2000                     # 最大训练轮数
    # -----------------------------------

    print("加载数据集...")
    # 获取数据集
    data = get_data(dataset_name, feat_dir, train_type, test_type)
    train = torch.from_numpy(data.train.x)
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(data, batch_size)
    
    print("初始化MADE模型...")
    # 获取模型
    n_in = data.n_dims  # 输入维度
    model = MADE(
        n_in, 
        hidden_dims, 
        random_order=random_order, 
        seed=seed, 
        gaussian=True,  # 使用高斯输出
        cuda_device=cuda_device
    )

    # 获取优化器
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    # 设置CUDA设备
    if cuda_device != None:
        torch.cuda.set_device(cuda_device)
        model = model.cuda()

    # 格式化模型保存文件名
    save_name = f"{model_name}_{dataset_name}_{train_type}_{'_'.join(str(d) for d in hidden_dims)}.pt"
    print(f"模型将保存为: {save_name}")
    
    # 初始化绘图列表
    epochs_list = []
    train_losses = []
    val_losses = []
    
    # 初始化早停机制
    i = 0                    # 耐心计数器
    max_loss = np.inf        # 最大损失值
    
    print("开始训练循环...")
    # 训练循环
    for epoch in range(1, max_epochs):
        # 训练一个epoch
        train_loss = train_one_epoch_made(model, epoch, optimiser, train_loader, cuda_device)
        
        # 验证
        val_loss = val_made(model, val_loader, cuda_device)

        # 记录训练历史
        epochs_list.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 早停机制：在每个epoch保存改进的模型
        if val_loss < max_loss and train_loss > min_loss:
            i = 0  # 重置耐心计数器
            max_loss = val_loss  # 更新最大损失值
            
            # 保存模型
            model = model.cpu()
            torch.save(
                model, os.path.join(model_dir, save_name)
            )  # 第一个epoch会打印UserWarning
            print(f"Epoch {epoch}: 验证损失改善，保存模型")
            
            # 恢复GPU训练
            if cuda_device != None:
                model = model.cuda()
        else:
            i += 1  # 增加耐心计数器

        # 打印耐心计数器状态
        if i < patience:
            print(f"耐心计数器: {i}/{patience}")
        else:
            print(f"耐心计数器: {i}/{patience}\n 终止训练！")
            break
    
    print("训练完成！")
    print(f"最佳验证损失: {max_loss:.6f}")
    print(f"总训练轮数: {epoch}")
