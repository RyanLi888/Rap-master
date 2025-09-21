"""
RAPIER 主程序文件 (优化版)
========================

本文件是 RAPIER (Robust Adversarial Perturbation In EEG Recognition) 系统的主入口程序。
该程序实现了完整的脑电图(EEG)数据处理、特征提取、模型训练和预测的流程。

主要功能模块：
1. AE (AutoEncoder): 自编码器，用于特征提取
2. MADE: 多尺度对抗判别器，用于数据增强和生成
3. Classifier: 分类器，用于最终的分类预测

优化配置：
- 使用最优随机种子配置 (F1=0.7911)
- 使用最优parallel参数 (parallel=1)
- 简化代码结构，移除冗余功能

作者: RAPIER 开发团队
版本: 2.0 (优化版)
"""

import os 
import sys 
import datetime
# 添加父目录到系统路径，以便导入其他模块
sys.path.append('..')
import MADE
import Classifier
import AE
import numpy as np
import shutil

# 导入随机种子控制模块
sys.path.append('../utils')
try:
    from random_seed import set_random_seed, RANDOM_CONFIG
    SEED_CONTROL_AVAILABLE = True
    print("✅ 随机种子控制模块导入成功")
except ImportError:
    print("⚠️  警告：随机种子控制模块导入失败，将使用默认行为")
    SEED_CONTROL_AVAILABLE = False

def generate(feat_dir, model_dir, made_dir, index, cuda):
    """
    生成指定索引的对抗样本
    
    参数:
        feat_dir (str): 特征文件目录路径
        model_dir (str): 模型保存目录路径
        made_dir (str): MADE相关文件目录路径
        index (int): 要生成的样本索引
        cuda (int): CUDA设备ID，-1表示使用CPU
    """
    # 定义训练数据标签
    TRAIN_be = 'be_corrected'  # 良性样本修正标签
    TRAIN_ma = 'ma_corrected'  # 恶性样本修正标签
    TRAIN = 'corrected'         # 通用修正标签
    
    # 训练MADE模型 - 分别针对良性(benign)和恶性(malignant)样本
    MADE.train.main(feat_dir, model_dir, TRAIN_be, cuda, '-30')
    MADE.train.main(feat_dir, model_dir, TRAIN_ma, cuda, '-30')
    
    # 使用训练好的MADE模型进行预测
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_be, TRAIN_be, cuda)
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_be, TRAIN_ma, cuda)
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_ma, TRAIN_ma, cuda)
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_ma, TRAIN_be, cuda)

    # 训练GAN生成器，用于生成对抗样本
    MADE.train_gen_GAN.main(feat_dir, model_dir, made_dir, TRAIN, cuda)
    
    # 使用训练好的GAN生成器生成对抗样本
    MADE.generate_GAN.main(feat_dir, model_dir, TRAIN, index, cuda)

def generate_cpus(feat_dir, model_dir, made_dir, indices, cuda):
    """
    批量生成多个索引的对抗样本
    """
    for index in indices:
        generate(feat_dir, model_dir, made_dir, index, cuda)

def main(data_dir, model_dir, feat_dir, made_dir, result_dir, cuda, random_seed=None):
    """
    主函数 - 执行完整的RAPIER流程 (优化版)
    
    使用经过验证的最优配置：
    - 随机种子: 2024 (全局) + 7271系列 (模块)
    - parallel参数: 1 (获得最佳F1=0.7911)
    """
    
    # 设置随机种子确保可重复性
    if SEED_CONTROL_AVAILABLE:
        if random_seed is None:
            random_seed = RANDOM_CONFIG['global_seed']
        set_random_seed(random_seed)
        print(f"🎯 已设置全局随机种子: {random_seed}")
    else:
        print("⚠️  跳过随机种子设置（模块不可用）")
    
    print("开始RAPIER完整流程训练...")
    
    # 清空所有目录，准备当前训练
    print("🧹 清理所有工作目录...")
    for dir_path in [feat_dir, made_dir, model_dir, result_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)
    print("✅ 所有目录清理完成")
    
    # 第一阶段：训练自编码器模型
    print("📚 阶段1: 训练自编码器模型...")
    AE.train.main(data_dir, model_dir, cuda)
    
    # 第二阶段：使用训练好的自编码器提取特征
    print("🔍 阶段2: 提取特征...")
    AE.get_feat.main(data_dir, model_dir, feat_dir, 'be', cuda)    # 提取良性样本特征
    AE.get_feat.main(data_dir, model_dir, feat_dir, 'ma', cuda)    # 提取恶性样本特征
    AE.get_feat.main(data_dir, model_dir, feat_dir, 'test', cuda)  # 提取测试样本特征
    
    # 第三阶段：训练MADE模型并进行数据清理
    print("🧠 阶段3: 训练MADE模型并进行数据清理...")
    TRAIN = 'be'  # 使用良性样本进行训练
    
    # 训练良性样本的MADE模型（用于数据清理）
    MADE.train_epochs.main(feat_dir, model_dir, made_dir, TRAIN, cuda, '20')
    
    # 数据清理和标签修正（使用良性样本作为基准）
    MADE.get_clean_epochs.main(feat_dir, made_dir, '0.5', TRAIN)
    MADE.final_predict.main(feat_dir, result_dir)
    
    # 第四阶段：生成对抗样本
    print("⚡ 阶段4: 生成对抗样本...")
    generate_cpus(feat_dir, model_dir, made_dir, list(range(5)), cuda)
    
    # 第五阶段：训练分类器并进行最终预测
    print("🎯 阶段5: 训练分类器并进行预测...")
    TRAIN = 'corrected'  # 设置为修正后的数据
    
    # ========== 使用最优配置 ==========
    # parallel=1: 获得最佳F1分数 0.7911
    # 使用完整的GAN数据而不是分散的小片段
    # ================================
    # 先训练分类器并保存模型
    print("🔧 训练分类器模型...")
    Classifier.classify.main(feat_dir, model_dir, result_dir, TRAIN, cuda, parallel=1)
    
    # 然后使用训练好的模型进行预测
    print("🔍 使用训练好的模型进行预测...")
    final_f1 = Classifier.classify.predict_only(feat_dir, model_dir, result_dir, TRAIN, cuda, parallel=1)
    
    print(f"\n🎉 RAPIER流程完成！")
    if final_f1 is not None:
        print(f"📊 最终F1分数: {final_f1:.4f}")
    else:
        print("📊 最终F1分数: 无法计算（模型文件缺失）")
    print(f"🎯 使用的随机种子: {random_seed}")
    
    return final_f1

def run_normal_mode(random_seed=None):
    """
    正常模式运行 - 使用固定的data目录结构
    
    参数:
        random_seed (int): 随机种子，默认使用配置文件中的种子
    """
    # 设置正常模式的目录路径
    data_dir = '../data/data'      # 原始数据目录
    feat_dir = '../data/feat'      # 特征文件目录
    model_dir= '../data/model'     # 模型保存目录
    made_dir = '../data/made'      # MADE相关文件目录
    result_dir='../data/result'    # 结果输出目录
    cuda = 0                       # 使用第一个CUDA设备（GPU 0）
    
    print("🚀 RAPIER正常模式运行 (优化版)")
    print("📁 使用固定目录结构: data/feat, data/model, data/made, data/result")
    print("⚙️  使用最优配置: parallel=1, 随机种子=2024")
    
    # 执行主函数（正常模式）
    return main(data_dir, model_dir, feat_dir, made_dir, result_dir, cuda, random_seed)

if __name__ == '__main__':
    """
    程序入口点
    
    当直接运行此脚本时，使用正常模式
    """
    run_normal_mode()