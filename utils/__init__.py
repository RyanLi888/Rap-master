"""
RAPIER 工具模块
===============

本包提供了RAPIER系统的各种工具函数，包括：
- 随机种子控制
- 数据处理工具
- 实验验证工具

作者: RAPIER 开发团队
版本: 1.0
"""

from .random_seed import set_random_seed, RANDOM_CONFIG

__all__ = ['set_random_seed', 'RANDOM_CONFIG']
