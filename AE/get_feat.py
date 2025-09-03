"""
AE (AutoEncoder) 特征提取模块
============================

本文件实现了使用训练好的自编码器模型提取特征的功能，包括：
1. 加载预训练的自编码器模型
2. 批量处理测试数据
3. 提取编码特征
4. 保存特征文件

作者: RAPIER 开发团队
版本: 1.0
"""

import sys
from .model import LSTM_AE_GMM
import numpy as np
import torch
import torch.nn as nn
import sys
import os

# 批次处理大小
batch_size = 128

def main(data_dir, model_dir, feat_dir, data_type, device):
    """
    特征提取主函数
    
    该函数使用训练好的自编码器模型，从原始时间序列数据中提取高维特征表示。
    提取的特征将用于后续的MADE模型训练和分类器训练。
    
    参数:
        data_dir (str): 原始数据目录路径
        model_dir (str): 预训练模型目录路径
        feat_dir (str): 特征保存目录路径
        data_type (str): 数据类型标识（'be', 'ma', 'test'）
        device (int/str): CUDA设备ID，'None'表示使用CPU
    """
    
    # 处理设备参数
    device = int(device) if device != 'None' else None
    
    print(f"开始处理 {data_type} 类型数据...")
    
    # 加载原始测试数据（包含标签）
    test_data_label = np.load(os.path.join(data_dir, data_type + '.npy'))
    
    # 分离特征和标签
    test_data = test_data_label[:, :50]      # 前50维为时间序列特征
    test_label = test_data_label[:, -1]      # 最后一维为标签
    
    # 获取数据总数
    total_size, _ = test_data.shape
    print(f"数据总数: {total_size}")

    # 设置CUDA设备
    device_id = int(device)
    torch.cuda.set_device(device_id)
    
    print("加载预训练的自编码器模型...")
    # 加载训练好的自编码器模型
    dagmm = torch.load(os.path.join(model_dir, 'gru_ae.pkl'))
    dagmm.to_cuda(device_id)
    dagmm = dagmm.cuda()
    dagmm.test_mode()  # 设置为测试模式
    
    print("开始特征提取...")
    # 批量提取特征
    feature = []
    for batch in range(total_size // batch_size + 1):
        # 检查是否超出数据范围
        if batch * batch_size >= total_size:
            break
            
        # 获取当前批次数据
        input = test_data[batch_size * batch : batch_size * (batch + 1)]
        
        # 使用自编码器提取特征（仅编码器部分）
        output = dagmm.feature(torch.Tensor(input).long().cuda())
        
        # 将特征转移到CPU并添加到列表
        feature.append(output.detach().cpu())

    # 合并所有批次的特征
    feature = torch.cat(feature, dim=0).numpy()
    
    # 将标签重新附加到特征上
    feature = np.concatenate([feature, test_label[:, None]], axis=1)
    
    # 保存特征文件
    output_path = os.path.join(feat_dir, data_type + '.npy')
    np.save(output_path, feature)
    print(f"特征已保存到: {output_path}")
    print(f"特征维度: {feature.shape}")