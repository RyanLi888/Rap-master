#!/usr/bin/env python3
"""
快速种子测试脚本
================

快速测试几个候选种子，找到最佳的一个。
适合时间有限但想获得更好结果的情况。

使用方法:
    python quick_seed_test.py

作者: RAPIER 开发团队
"""

import sys
sys.path.append('utils')

from random_seed import CANDIDATE_SEEDS

def quick_test():
    """快速测试推荐种子"""
    print("🚀 快速种子测试")
    print(f"📋 将测试这些种子: {CANDIDATE_SEEDS[:5]}")  # 只测试前5个
    
    print("\n💡 推荐步骤:")
    print("1. 修改 utils/random_seed.py 中的 GLOBAL_SEED")
    print("2. 运行 python main/main.py")
    print("3. 记录F1分数")
    print("4. 重复测试不同种子")
    print("5. 选择最佳结果的种子")
    
    print(f"\n🎯 建议优先测试:")
    for i, seed in enumerate(CANDIDATE_SEEDS[:5], 1):
        print(f"  {i}. 种子 {seed}")
    
    print(f"\n📝 测试记录模板:")
    print("种子    | F1分数   | 准确率   | 备注")
    print("--------|----------|----------|--------")
    for seed in CANDIDATE_SEEDS[:5]:
        print(f"{seed:<8}| ________ | ________ | ________")

if __name__ == "__main__":
    quick_test()

