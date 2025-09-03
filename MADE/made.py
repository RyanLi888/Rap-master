"""
MADE (Masked Autoencoder for Distribution Estimation) 模型
=======================================================

本文件实现了MADE模型，这是一种用于密度估计的掩码自编码器。
该模型通过掩码机制确保自回归性质，能够有效建模高维数据的条件分布。

主要特性：
1. 掩码线性层：确保自回归性质
2. 支持高斯和伯努利输出
3. 可配置的隐藏层维度
4. 支持随机输入排序
5. GPU加速支持

参考实现: https://github.com/e-hulten/made

作者: RAPIER 开发团队
版本: 1.0
"""

from typing import List, Optional
import numpy as np
from numpy.random import permutation, randint
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import ReLU

# This implementation of MADE is copied from: https://github.com/e-hulten/made.

class MaskedLinear(nn.Linear):
    """
    带掩码的线性变换层
    
    通过掩码机制限制权重矩阵的连接，确保自回归性质。
    公式: y = x.dot(mask*W.T) + b
    """

    def __init__(self, n_in: int, n_out: int, bias: bool = True, cuda_device: Optional[int] = None) -> None:
        """
        初始化掩码线性层
        
        参数:
            n_in: 输入样本的维度
            n_out: 输出样本的维度
            bias: 是否包含偏置项，默认True
            cuda_device: CUDA设备ID，None表示使用CPU
        """
        super().__init__(n_in, n_out, bias)
        self.mask = None
        self.cuda_device = cuda_device

    def initialise_mask(self, mask: Tensor):
        """
        内部方法：初始化掩码
        
        参数:
            mask: 掩码张量
        """
        if self.cuda_device == None:
            self.mask = mask
        else:
            torch.cuda.set_device(self.cuda_device)
            self.mask = mask.cuda()
            
    def set_device(self, device):
        """
        设置CUDA设备
        
        参数:
            device: 目标CUDA设备ID
        """
        torch.cuda.set_device(device)
        self.cuda_device = device
        self.mask = self.mask.cpu().cuda()
        
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播：应用掩码线性变换
        
        参数:
            x: 输入张量
            
        返回:
            掩码后的线性变换结果
        """
        return F.linear(x, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    """
    MADE (Masked Autoencoder for Distribution Estimation) 主模型
    
    该模型通过掩码机制确保自回归性质，能够有效建模高维数据的条件分布。
    支持高斯和伯努利输出，适用于密度估计任务。
    """
    
    def __init__(
        self,
        n_in: int,
        hidden_dims: List[int],
        gaussian: bool = False,
        random_order: bool = False,
        seed: Optional[int] = None,
        cuda_device: Optional[int] = None,
    ) -> None:
        """
        初始化MADE模型
    
        参数:
            n_in: 输入维度
            hidden_dims: 隐藏层维度列表
            gaussian: 是否使用高斯MADE（输出mu和sigma），默认False
            random_order: 是否使用随机输入排序，默认False
            seed: numpy随机种子，默认None
            cuda_device: CUDA设备ID，None表示使用CPU
        """
        super().__init__()
        
        # 设置随机种子
        np.random.seed(seed)
        
        # 模型参数
        self.n_in = n_in
        self.n_out = 2 * n_in if gaussian else n_in  # 高斯输出维度翻倍
        self.hidden_dims = hidden_dims
        self.random_order = random_order
        self.gaussian = gaussian
        self.masks = {}
        self.mask_matrix = []
        self.layers = []
        self.cuda_device = cuda_device

        # 设置CUDA设备
        if self.cuda_device != None:
            torch.cuda.set_device(self.cuda_device)

        # 构建层维度列表
        dim_list = [self.n_in, *hidden_dims, self.n_out]

        # 创建隐藏层和激活函数
        for i in range(len(dim_list) - 2):
            if self.cuda_device == None:
                self.layers.append(MaskedLinear(dim_list[i], dim_list[i + 1]))
                self.layers.append(ReLU())
            else:
                self.layers.append(MaskedLinear(dim_list[i], dim_list[i + 1], 
                    cuda_device=self.cuda_device).cuda())
                self.layers.append(ReLU().cuda())

        # 最后一层：隐藏层到输出层
        if self.cuda_device == None:
            self.layers.append(MaskedLinear(dim_list[-2], dim_list[-1]))
        else:
            self.layers.append(MaskedLinear(dim_list[-2], dim_list[-1], 
                cuda_device=self.cuda_device).cuda())

        # 创建顺序模型
        self.model = nn.Sequential(*self.layers)

        # 为掩码激活创建掩码
        self._create_masks()

    def set_device(self, device):
        """
        设置CUDA设备
        
        参数:
            device: 目标CUDA设备ID
        """
        torch.cuda.set_device(device)
        self.cuda_device = device
        for model in self.model:
            if isinstance(model, MaskedLinear):
                model.set_device(device)
            model = model.cpu().cuda()
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量
            
        返回:
            模型输出（高斯或伯努利）
        """
        if self.gaussian:
            # 高斯输出：返回原始的mu和sigma
            res = self.model(x)
            return res
        else:
            # 伯努利输出：通过sigmoid将概率压缩到(0,1)区间
            return torch.sigmoid(self.model(x))

    def _create_masks(self) -> None:
        """
        为隐藏层创建掩码
        
        该方法实现了MADE的核心机制：通过掩码确保自回归性质。
        每个隐藏单元只能连接到输入层中索引小于等于其连接数的单元。
        """
        # 定义常量以提高可读性
        L = len(self.hidden_dims)  # 隐藏层数量
        D = self.n_in              # 输入维度

        # 决定是否使用随机或自然输入排序
        self.masks[0] = permutation(D) if self.random_order else np.arange(D)

        # 为隐藏层设置连接数m
        # m ~ DiscreteUniform[min_{prev_layer}(m), D-1]
        for l in range(L):
            low = self.masks[l].min()
            size = self.hidden_dims[l]
            self.masks[l + 1] = randint(low=low, high=D - 1, size=size)

        # 为输出层添加m，输出顺序与输入顺序相同
        self.masks[L + 1] = self.masks[0]

        # 为输入->隐藏1->...->隐藏L创建掩码矩阵
        for i in range(len(self.masks) - 1):
            m = self.masks[i]
            m_next = self.masks[i + 1]
            
            # 初始化掩码矩阵
            M = torch.zeros(len(m_next), len(m))
            for j in range(len(m_next)):
                # 使用广播比较m_next[j]与m中的每个元素
                M[j, :] = torch.from_numpy((m_next[j] >= m).astype(int))
            
            # 添加到掩码矩阵列表
            self.mask_matrix.append(M)

        # 如果输出是高斯分布，将输出单元数量翻倍（mu, sigma）
        # 成对相同的掩码
        if self.gaussian:
            m = self.mask_matrix.pop(-1)
            self.mask_matrix.append(torch.cat((m, m), dim=0))

        # 用权重初始化MaskedLinear层
        mask_iter = iter(self.mask_matrix)
        for module in self.model.modules():
            if isinstance(module, MaskedLinear):
                module.initialise_mask(next(mask_iter))
