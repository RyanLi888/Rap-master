"""
MADE 数据加载器模块
==================

本文件提供了数据集的加载和预处理功能，包括：
1. get_data: 根据数据集名称返回对应的数据集对象
2. get_data_loaders: 创建训练、验证和测试数据加载器

作者: RAPIER 开发团队
版本: 1.0
"""

import torch
from .myData import MyDataset

def get_data(dataset: str, feat_dir=None, train_type=None, test_type=None):
    """
    根据数据集名称获取对应的数据集对象
    
    参数:
        dataset (str): 数据集名称，目前支持 'myData'
        feat_dir (str): 特征文件目录路径
        train_type (str): 训练数据类型标识
        test_type (str): 测试数据类型标识
        
    返回:
        MyDataset: 数据集对象
        
    异常:
        ValueError: 当数据集名称不支持时抛出
    """
    if dataset == 'myData':
        return MyDataset(feat_dir, train_type, test_type)

    raise ValueError(
        f"Unknown dataset '{dataset}'. Please choose either 'mnist', 'power', or 'hepmass'."
    )

def get_data_loaders(data, batch_size: int = 1024):
    """
    创建训练、验证和测试数据加载器
    
    参数:
        data: 数据集对象，包含 train、val、test 属性
        batch_size (int): 批次大小，默认1024
        
    返回:
        tuple: (训练加载器, 验证加载器, 测试加载器)
    """
    train = torch.from_numpy(data.train.x)
    val = torch.from_numpy(data.val.x)
    test = torch.from_numpy(data.test.x)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size,)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,)

    return train_loader, val_loader, test_loader
