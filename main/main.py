"""
RAPIER 主程序文件
================

本文件是 RAPIER (Robust Adversarial Perturbation In EEG Recognition) 系统的主入口程序。
该程序实现了完整的脑电图(EEG)数据处理、特征提取、模型训练和预测的流程。

主要功能模块：
1. AE (AutoEncoder): 自编码器，用于特征提取
2. MADE: 多尺度对抗判别器，用于数据增强和生成
3. Classifier: 分类器，用于最终的分类预测

作者: RAPIER 开发团队
版本: 1.0
"""

import os 
import sys 
# 添加父目录到系统路径，以便导入其他模块
sys.path.append('..')
import MADE
import Classifier
import AE

def generate(feat_dir, model_dir, made_dir, index, cuda):
    """
    生成指定索引的对抗样本
    
    该函数使用训练好的MADE模型生成对抗样本，包括：
    1. 训练MADE模型（分别针对be和ma数据）
    2. 使用训练好的模型进行预测
    3. 训练GAN生成器
    4. 生成最终的对抗样本
    
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
    # 参数格式: (特征目录, 模型目录, MADE目录, 训练标签, 预测标签, CUDA设备)
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_be, TRAIN_be, cuda)  # 良性->良性
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_be, TRAIN_ma, cuda)  # 良性->恶性
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_ma, TRAIN_ma, cuda)  # 恶性->恶性
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_ma, TRAIN_be, cuda)  # 恶性->良性

    # 训练GAN生成器，用于生成对抗样本
    MADE.train_gen_GAN.main(feat_dir, model_dir, made_dir, TRAIN, cuda)
    # 使用训练好的GAN生成器生成对抗样本
    MADE.generate_GAN.main(feat_dir, model_dir, TRAIN, index, cuda)

def generate_cpus(feat_dir, model_dir, made_dir, indices, cuda):
    """
    批量生成多个索引的对抗样本
    
    该函数循环调用generate函数，为指定的多个索引生成对抗样本。
    
    参数:
        feat_dir (str): 特征文件目录路径
        model_dir (str): 模型保存目录路径
        made_dir (str): MADE相关文件目录路径
        indices (list): 要生成的样本索引列表
        cuda (int): CUDA设备ID
    """
    for index in indices:
        generate(feat_dir, model_dir, made_dir, index, cuda)

def main(data_dir, model_dir, feat_dir, made_dir, result_dir, cuda):
    """
    主函数 - 执行完整的RAPIER流程
    
    该函数实现了RAPIER系统的完整工作流程：
    1. 训练自编码器(AE)模型
    2. 提取特征（良性、恶性、测试数据）
    3. 训练MADE模型并进行数据清理
    4. 生成对抗样本
    5. 训练分类器并进行最终预测
    
    参数:
        data_dir (str): 原始数据目录路径
        model_dir (str): 模型保存目录路径
        feat_dir (str): 特征文件目录路径
        made_dir (str): MADE相关文件目录路径
        result_dir (str): 结果输出目录路径
        cuda (int): CUDA设备ID
    """
    
    # 第一步：训练自编码器模型
    print("开始训练自编码器模型...")
    AE.train.main(data_dir, model_dir, cuda)
    
    # 第二步：使用训练好的自编码器提取特征
    print("开始提取特征...")
    AE.get_feat.main(data_dir, model_dir, feat_dir, 'be', cuda)    # 提取良性样本特征
    AE.get_feat.main(data_dir, model_dir, feat_dir, 'ma', cuda)    # 提取恶性样本特征
    AE.get_feat.main(data_dir, model_dir, feat_dir, 'test', cuda)  # 提取测试样本特征

    # 第三步：训练MADE模型并进行数据清理
    print("开始训练MADE模型...")
    TRAIN = 'be'  # 使用良性样本进行训练
    MADE.train_epochs.main(feat_dir, model_dir, made_dir, TRAIN, cuda, '20')  # 训练20个epoch
    MADE.get_clean_epochs.main(feat_dir, made_dir, '0.5', TRAIN)  # 获取清理后的epoch，阈值0.5
    MADE.final_predict.main(feat_dir)  # 进行最终预测
    
    # 第四步：生成对抗样本（为5个不同的索引生成）
    print("开始生成对抗样本...")
    generate_cpus(feat_dir, model_dir, made_dir, list(range(5)), cuda)
    
    # 第五步：训练分类器并进行最终分类
    print("开始训练分类器...")
    TRAIN = 'corrected'  # 使用修正后的数据进行训练
    Classifier.classify.main(feat_dir, model_dir, result_dir, TRAIN, cuda, parallel=5)  # 使用5个并行进程
    
    print("RAPIER流程完成！")

if __name__ == '__main__':
    """
    程序入口点
    
    当直接运行此脚本时，设置默认参数并执行主函数
    """
    # 设置默认的目录路径
    data_dir = '../data/data'      # 原始数据目录
    feat_dir = '../data/feat'      # 特征文件目录
    model_dir= '../data/model'     # 模型保存目录
    made_dir = '../data/made'      # MADE相关文件目录
    result_dir='../data/result'    # 结果输出目录
    cuda = 0                       # 使用第一个CUDA设备（GPU 0）
    
    # 执行主函数
    main(data_dir, model_dir, feat_dir, made_dir, result_dir, cuda)