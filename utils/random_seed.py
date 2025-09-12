"""
随机种子控制模块
==================

本模块提供了统一的随机种子设置功能，确保实验结果的可重复性。
通过设置所有相关库的随机种子，可以使RAPIER系统的每次运行产生相同的结果。

主要功能：
1. 设置Python内置random模块的种子
2. 设置NumPy随机种子
3. 设置PyTorch随机种子（CPU和GPU）
4. 设置CUDA随机种子
5. 控制PyTorch的确定性行为

作者: RAPIER 开发团队
版本: 1.0
"""

import random
import numpy as np
import torch
import os


def set_random_seed(seed=42):
    """
    设置所有相关库的随机种子，确保实验可重复性
    
    参数:
        seed (int): 随机种子值，默认为42
    """
    print(f"🎯 设置随机种子为: {seed}")
    
    # 1. 设置Python内置random模块的种子
    random.seed(seed)
    
    # 2. 设置NumPy随机种子
    np.random.seed(seed)
    
    # 3. 设置PyTorch随机种子
    torch.manual_seed(seed)
    
    # 4. 设置CUDA随机种子（如果有GPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况
        
        # 5. 设置CUDA确定性行为
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        print("✅ GPU随机种子已设置，启用确定性模式")
    
    # 6. 设置环境变量以确保完全确定性
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print("✅ 所有随机种子设置完成")


def create_deterministic_dataloader(dataset, batch_size, shuffle=True, seed=42):
    """
    创建确定性的数据加载器
    
    参数:
        dataset: PyTorch数据集
        batch_size (int): 批次大小
        shuffle (bool): 是否打乱数据
        seed (int): 随机种子
        
    返回:
        torch.utils.data.DataLoader: 确定性数据加载器
    """
    if shuffle:
        # 创建确定性的随机采样器
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
            worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
    
    return dataloader


def deterministic_shuffle(array, seed=42):
    """
    确定性的数组打乱函数
    
    参数:
        array (np.ndarray): 要打乱的数组
        seed (int): 随机种子
        
    返回:
        np.ndarray: 打乱后的数组
    """
    # 临时设置种子
    state = np.random.get_state()
    np.random.seed(seed)
    
    # 打乱数组
    shuffled_array = array.copy()
    np.random.shuffle(shuffled_array)
    
    # 恢复原始随机状态
    np.random.set_state(state)
    
    return shuffled_array


def get_deterministic_random_int(low, high, seed=42):
    """
    生成确定性的随机整数
    
    参数:
        low (int): 最小值
        high (int): 最大值（不包含）
        seed (int): 随机种子
        
    返回:
        int: 确定性的随机整数
    """
    # 创建独立的随机数生成器
    rng = np.random.RandomState(seed)
    return rng.randint(low, high)


class DeterministicContext:
    """
    确定性上下文管理器
    
    在with块内确保所有操作都是确定性的
    """
    
    def __init__(self, seed=42):
        self.seed = seed
        self.original_state = None
        self.original_torch_state = None
        
    def __enter__(self):
        # 保存当前状态
        self.original_state = np.random.get_state()
        self.original_torch_state = torch.get_rng_state()
        
        # 设置确定性种子
        set_random_seed(self.seed)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复原始状态
        np.random.set_state(self.original_state)
        torch.set_rng_state(self.original_torch_state)


def verify_reproducibility(func, *args, **kwargs):
    """
    验证函数的可重复性
    
    参数:
        func: 要验证的函数
        *args: 函数的位置参数
        **kwargs: 函数的关键字参数
        
    返回:
        bool: 是否可重复
    """
    print("🔍 验证可重复性...")
    
    # 第一次运行
    set_random_seed(42)
    result1 = func(*args, **kwargs)
    
    # 第二次运行
    set_random_seed(42)
    result2 = func(*args, **kwargs)
    
    # 比较结果
    if isinstance(result1, torch.Tensor):
        is_reproducible = torch.allclose(result1, result2, atol=1e-6)
    elif isinstance(result1, np.ndarray):
        is_reproducible = np.allclose(result1, result2, atol=1e-6)
    else:
        is_reproducible = (result1 == result2)
    
    if is_reproducible:
        print("✅ 函数具有可重复性")
    else:
        print("❌ 函数不具有可重复性")
    
    return is_reproducible


# 预定义的种子值 - 可以修改这些值来尝试不同组合
GLOBAL_SEED = 42        # 全局种子 - 修改这个值来尝试不同结果
AE_SEED = 290713       # 使用MADE中的原始种子
MADE_SEED = 290713     # MADE模型种子
CLASSIFIER_SEED = 12345 # 分类器种子
GENERATION_SEED = 54321 # 生成器种子

# 常用的高性能种子候选（基于经验）
CANDIDATE_SEEDS = [
    42,      # 经典种子
    123,     # 简单序列
    290713,  # 原MADE种子
    291713,  # MADE种子变体
    292713,  # MADE种子变体
    1234,    # 常用种子
    2024,    # 年份种子
    12345,   # 递增序列
    54321,   # 递减序列
    99999    # 大数值种子
]

# 导出的配置
RANDOM_CONFIG = {
    'global_seed': GLOBAL_SEED,
    'ae_seed': AE_SEED, 
    'made_seed': MADE_SEED,
    'classifier_seed': CLASSIFIER_SEED,
    'generation_seed': GENERATION_SEED
}

if __name__ == "__main__":
    # 测试随机种子设置
    print("🧪 测试随机种子控制模块...")
    
    set_random_seed(42)
    
    # 测试NumPy随机性
    print(f"NumPy随机数: {np.random.random()}")
    
    # 测试PyTorch随机性
    print(f"PyTorch随机数: {torch.rand(1).item()}")
    
    print("✅ 随机种子控制模块测试完成")
