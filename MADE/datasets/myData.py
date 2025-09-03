"""
MADE 自定义数据集模块
====================

本文件实现了自定义数据集类，用于加载和预处理特征数据。
主要功能包括数据加载、标准化和训练/验证/测试集划分。

作者: RAPIER 开发团队
版本: 1.0
"""

import numpy as np
import os

class MyDataset:
    """
    自定义数据集类
    
    用于加载和预处理特征数据，支持训练集、验证集和测试集的划分。
    """
    
    class Data:
        """
        内部数据类
        
        包装numpy数组并提供样本数量信息。
        """
        def __init__(self, data):
            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]
    
    def __init__(self, feat_dir, train_type, test_type):
        """
        初始化数据集
        
        参数:
            feat_dir (str): 特征文件目录路径
            train_type (str): 训练数据类型标识
            test_type (str): 测试数据类型标识
        """
        
        train_file = os.path.join(feat_dir, train_type + '.npy')
        test_file = os.path.join(feat_dir, test_type + '.npy')

        # 加载并标准化数据
        train, valid, test = load_data_normalized(train_file, test_file)

        self.train = self.Data(train)
        self.val = self.Data(valid)
        self.test = self.Data(test)

        self.n_dims = self.train.x.shape[1]

def load_data(root_path, is_train):
    """
    加载原始数据文件
    
    参数:
        root_path (str): 数据文件路径
        is_train (bool): 是否为训练数据（影响是否划分验证集）
        
    返回:
        tuple or np.ndarray: 训练数据返回(train, valid)元组，测试数据返回单个数组
    """
    data = np.load(root_path)[:, :32]  # 只使用前32维特征
    
    if is_train is True:
        # 训练数据：划分10%作为验证集
        N_validate = int(0.1 * data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data
        return data_train, data_validate
    else:
        # 测试数据：直接返回
        return data

def load_data_normalized(train_path, test_path):
    """
    加载并标准化数据
    
    使用训练集的均值和标准差对所有数据进行标准化处理。
    
    参数:
        train_path (str): 训练数据文件路径
        test_path (str): 测试数据文件路径
        
    返回:
        tuple: (标准化后的训练集, 验证集, 测试集)
    """
    data_train, data_validate = load_data(train_path, True)
    data_test = load_data(test_path, False)
    data = data_train

    # 计算训练集的统计量
    mu = data.mean(axis=0)
    s = data.std(axis=0)

    # 处理标准差为0的情况（避免除零错误）
    for i, score in enumerate(s):
        if score == 0:
            s[i] = 1

    # 标准化所有数据
    data_train = (data_train - mu) / s
    data_validate = (data_validate - mu) / s
    data_test = (data_test - mu) / s

    return data_train, data_validate, data_test
