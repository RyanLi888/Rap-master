"""
MADE (Multi-scale Adversarial Discriminator) 多尺度对抗判别器模块
================================================================

本模块实现了多尺度对抗判别器功能，用于脑电图数据的对抗样本生成和数据增强。

主要组件：
- train_epochs: 分轮次训练模块
- get_clean_epochs: 获取清理后的epoch数据
- final_predict: 最终预测模块
- train: 基础训练模块
- predict: 预测模块
- train_gen_GAN: GAN生成器训练模块
- generate_GAN: GAN样本生成模块

作者: RAPIER 开发团队
版本: 1.0
"""

from . import train_epochs
from . import get_clean_epochs
from . import final_predict
from . import train
from . import predict
from . import train_gen_GAN
from . import generate_GAN
