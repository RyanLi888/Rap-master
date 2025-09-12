"""
使用训练好的生成器批量生成增强样本
=============================

从已训练的三类生成器（be/ma1/ma2）中采样，按对应训练集的统计量进行尺度还原，
并将生成的特征保存到磁盘。

作者: RAPIER 开发团队
版本: 1.0
"""

from .gen_model import GEN
from .made import MADE
import torch
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F
import sys
import os 
from sklearn.datasets import make_blobs
import math

# 导入随机种子控制模块
sys.path.append('../utils')
try:
    from random_seed import get_deterministic_random_int, RANDOM_CONFIG
    SEED_CONTROL_AVAILABLE = True
except ImportError:
    SEED_CONTROL_AVAILABLE = False

# 合成3类样本
def main(feat_dir, model_dir, TRAIN, index, cuda_device):
    """
    使用已训练的生成器导出增强样本
    
    参数:
        feat_dir (str): 特征目录
        model_dir (str): 模型目录（包含生成器）
        TRAIN (str): 训练标签后缀，例如 'corrected'
        index (int): 当前生成批次编号
        cuda_device (int|str): CUDA设备ID
    """

    be = np.load(os.path.join(feat_dir, 'be_%s.npy'%(TRAIN)))[:, :32]
    ma = np.load(os.path.join(feat_dir, 'ma_%s.npy'%(TRAIN)))[:, :32]

    output_size = be.shape[1]
    hiddens = [8, 16]
    device = int(cuda_device) if cuda_device != 'None' else None
    train_type_be = 'be_' + TRAIN
    train_type_ma = 'ma_' + TRAIN

    load_name_be = f"gen_GAN_{train_type_be}_{'_'.join(str(d) for d in hiddens)}.pt"
    load_name_ma1 = f"gen1_GAN_{train_type_ma}_{'_'.join(str(d) for d in hiddens)}.pt"
    load_name_ma2 = f"gen2_GAN_{train_type_ma}_{'_'.join(str(d) for d in hiddens)}.pt"
    BeGenModel = torch.load(os.path.join(model_dir, load_name_be))
    MaGenModel_1 = torch.load(os.path.join(model_dir, load_name_ma1))
    MaGenModel_2 = torch.load(os.path.join(model_dir, load_name_ma2))

    if device != None:
        torch.cuda.set_device(device)
        BeGenModel.to_cuda(device)
        BeGenModel = BeGenModel.cuda()
        MaGenModel_1.to_cuda(device)
        MaGenModel_1 = MaGenModel_1.cuda()
        MaGenModel_2.to_cuda(device)
        MaGenModel_2 = MaGenModel_2.cuda()

    def generate(train_type, GenModel, total_size, seed):
        """
        根据训练集统计量对生成样本做尺度还原，并返回numpy数组。
        """
        data_train = np.load(os.path.join(feat_dir, train_type + '.npy'))[:, :output_size]
        mu_train = torch.Tensor(data_train.mean(axis=0))
        s_train = torch.Tensor(data_train.std(axis=0))

        if device != None:
            mu_train = mu_train.cuda()
            s_train = s_train.cuda()

        X, _ = make_blobs(n_samples=total_size, centers=[[0, 0]], n_features=2, random_state=seed)
        X = torch.Tensor(X)
        batch = GenModel.forward(X)
        batch1 = batch * s_train + mu_train
        gen_data = batch1.detach().cpu()
        
        return np.array(gen_data)

    # 使用确定性随机种子
    if SEED_CONTROL_AVAILABLE:
        be_seed = get_deterministic_random_int(0, 1000, seed=RANDOM_CONFIG['generation_seed'] + index)
        ma1_seed = get_deterministic_random_int(0, 1000, seed=RANDOM_CONFIG['generation_seed'] + index + 1000)
        ma2_seed = get_deterministic_random_int(0, 1000, seed=RANDOM_CONFIG['generation_seed'] + index + 2000)
        print(f"✅ 生成器: 使用确定性种子 be={be_seed}, ma1={ma1_seed}, ma2={ma2_seed}")
    else:
        be_seed = np.random.randint(1000)
        ma1_seed = np.random.randint(1000)
        ma2_seed = np.random.randint(1000)
        print("⚠️  生成器: 使用非确定性种子")
    
    gen_data_be = generate(train_type_be, BeGenModel, int(be.shape[0]) * 2, be_seed)
    gen_data_ma1 = generate(train_type_ma, MaGenModel_1, int(ma.shape[0]) * 2, ma1_seed)
    gen_data_ma2 = generate(train_type_ma, MaGenModel_2, int(ma.shape[0]) * 2, ma2_seed)

    np.save(os.path.join(feat_dir, 'be_%s_generated_GAN_%d.npy'%(TRAIN, index)), gen_data_be)
    np.save(os.path.join(feat_dir, 'ma_%s_generated_GAN_1_%d.npy'%(TRAIN, index)), gen_data_ma1)
    np.save(os.path.join(feat_dir, 'ma_%s_generated_GAN_2_%d.npy'%(TRAIN, index)), gen_data_ma2)
