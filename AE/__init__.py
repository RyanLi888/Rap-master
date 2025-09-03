"""
AE (AutoEncoder) 自编码器模块
============================

本模块实现了自编码器功能，用于脑电图数据的特征提取和降维。

主要组件：
- train: 自编码器训练模块
- get_feat: 特征提取模块

作者: RAPIER 开发团队
版本: 1.0
"""

from . import train 
from . import get_feat