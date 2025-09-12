"""
AE (AutoEncoder) 自编码器训练模块
================================

本文件实现了自编码器模型的训练流程，包括：
1. 数据加载和预处理
2. 模型初始化和配置
3. 训练循环和优化
4. 模型保存

作者: RAPIER 开发团队
版本: 1.0
"""

from .model import LSTM_AE_GMM
import numpy as np
import torch
import torch.nn as nn
import sys
import os

# 导入随机种子控制模块
sys.path.append('../utils')
try:
    from random_seed import deterministic_shuffle, RANDOM_CONFIG
    SEED_CONTROL_AVAILABLE = True
except ImportError:
    SEED_CONTROL_AVAILABLE = False

# 训练超参数
batch_size = 128        # 批次大小
Max_epochs = 1000       # 最大训练轮数

def main(data_dir, model_dir, device):
    """
    自编码器训练主函数
    
    该函数实现了完整的自编码器训练流程，包括数据加载、模型训练和保存。
    
    参数:
        data_dir (str): 训练数据目录路径
        model_dir (str): 模型保存目录路径
        device (int): CUDA设备ID
    """
    
    print("开始加载训练数据...")
    # 加载原始时间序列训练数据
    train_data_be = np.load(os.path.join(data_dir, 'be.npy'))  # 良性样本数据
    train_data_ma = np.load(os.path.join(data_dir, 'ma.npy'))  # 恶性样本数据
    
    # 合并良性 and 恶性样本，只使用前50个特征维度（时间序列数据）
    train_data = np.concatenate([train_data_be[:, :50], train_data_ma[:, :50]], axis=0)
    
    # 使用确定性打乱数据顺序
    if SEED_CONTROL_AVAILABLE:
        train_data = deterministic_shuffle(train_data, seed=RANDOM_CONFIG['ae_seed'])
        print("✅ AE: 使用确定性数据打乱")
    else:
        np.random.shuffle(train_data)  # 原始随机打乱
        print("⚠️  AE: 使用非确定性数据打乱")
    
    # 备用数据加载方式（已注释）
    #train_data = np.load(os.path.join(data_dir, 'all.npy'))[:, :50]
    #print(train_data.shape)
    
    # 获取数据维度信息
    total_size, input_size = train_data.shape
    print(f"训练数据总数: {total_size}")
    
    # 设置CUDA设备
    device_id = int(device)
    print(f"使用CUDA设备: {device_id}")
    torch.cuda.set_device(device_id)

    # 根据数据量调整训练轮数
    max_epochs = Max_epochs * 200 // total_size 
    print(f"调整后的训练轮数: {max_epochs}")
    
    print("初始化自编码器模型...")
    # 创建LSTM自编码器模型实例
    dagmm = LSTM_AE_GMM(
        input_size=input_size,      # 输入序列长度
        max_len=2000,               # 最大序列长度
        emb_dim=32,                 # 词向量维度
        hidden_size=8,              # LSTM隐藏层维度
        dropout=0.2,                # Dropout比率
        est_hidden_size=64,         # 估计器隐藏层维度
        est_output_size=8,          # 估计器输出维度
        device=device_id,            # CUDA设备ID
    ).cuda()

    # 设置为训练模式
    dagmm.train_mode()
    
    # 初始化Adam优化器
    optimizer = torch.optim.Adam(dagmm.parameters(), lr=1e-2)
    
    print("开始训练...")
    # 训练循环
    for epoch in range(max_epochs):
        # 批次训练
        for batch in range(total_size // batch_size + 1):
            # 检查是否超出数据范围
            if batch * batch_size >= total_size:
                break
                
            # 清零梯度
            optimizer.zero_grad()
            
            # 获取当前批次数据
            input = train_data[batch_size * batch : batch_size * (batch + 1)]
            
            # 前向传播和损失计算
            loss = dagmm.loss(torch.Tensor(input).long().cuda())
            
            # 反向传播和参数更新
            loss.backward()
            optimizer.step()
        
        # 打印训练进度
        print(f'epoch: {epoch}, loss: {loss.item():.6f}')
        
        # 每50个epoch保存一次模型
        if (epoch + 1) % 50 == 0:
            print(f"保存模型到 {model_dir}/gru_ae.pkl")
            # 将模型转移到CPU并保存
            dagmm.to_cpu()
            dagmm = dagmm.cpu()
            torch.save(dagmm, os.path.join(model_dir, 'gru_ae.pkl'))
            # 恢复GPU训练
            dagmm.to_cuda(device_id)
            dagmm = dagmm.cuda()
    
    print("训练完成！")
