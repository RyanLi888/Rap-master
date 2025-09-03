"""
Classifier 分类器模型
===================

本文件实现了基本的MLP分类器模型，用于对提取的特征进行分类预测。
该模型采用多层感知机架构，支持GPU加速训练。

主要特性：
1. 可配置的隐藏层维度
2. 支持CPU和GPU训练
3. 使用Tanh激活函数
4. 灵活的设备切换

作者: RAPIER 开发团队
版本: 1.0
"""

import torch
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 

class MLP(nn.Module):
    """
    多层感知机分类器模型
    
    该模型采用全连接层架构，用于对提取的特征进行分类。
    支持可配置的隐藏层维度和GPU加速。
    
    参数:
        input_size (int): 输入特征维度
        hiddens (list): 隐藏层维度列表
        output_size (int): 输出类别数
        device (int): CUDA设备ID，None表示使用CPU
    """

    def __init__(self, input_size, hiddens, output_size, device=None):
        super().__init__()
        
        # 模型参数
        self.input_size = input_size
        self.output_size = output_size
        self.dim_list = [input_size, *hiddens, output_size]  # 构建完整的维度列表
        self.device = device

        # 设置CUDA设备
        if device != None:
            torch.cuda.set_device(device)

        # 构建网络层
        self.layers = []
        for (dim1, dim2) in zip(self.dim_list[:-2], self.dim_list[1:-1]):
            if self.device != None:
                # GPU版本：添加线性层和激活函数
                self.layers.append(nn.Linear(dim1, dim2).cuda())
                self.layers.append(nn.Tanh().cuda())
            else:
                # CPU版本：添加线性层和激活函数
                self.layers.append(nn.Linear(dim1, dim2))
                self.layers.append(nn.Tanh())
        
        # 添加最后一层（输出层）
        if self.device != None:
            self.layers.append(nn.Linear(self.dim_list[-2], self.dim_list[-1]).cuda())
        else:
            self.layers.append(nn.Linear(self.dim_list[-2], self.dim_list[-1]))
        
        # 创建顺序模型
        self.models = nn.Sequential(*self.layers)

    def forward(self, input):
        """
        前向传播
        
        参数:
            input (torch.Tensor): 输入特征张量
            
        返回:
            torch.Tensor: 分类输出
            
        异常:
            AssertionError: 当输入维度不匹配时抛出
        """
        # 验证输入维度
        assert(input.shape[1] == self.input_size)

        # 将输入转移到指定设备
        if self.device != None:
            torch.cuda.set_device(self.device)
            input = input.cuda()
        
        # 前向传播
        output = self.models(input)
        return output
    
    def to_cpu(self):
        """将模型转移到CPU"""
        self.device = None
        for model in self.models:
            model = model.cpu()
    
    def to_cuda(self, device):
        """
        将模型转移到指定GPU
        
        参数:
            device (int): 目标CUDA设备ID
        """
        self.device = device
        torch.cuda.set_device(self.device)
        for model in self.models:
            model = model.cuda()
