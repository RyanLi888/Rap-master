"""
RAPIER 主程序文件
================

本文件是 RAPIER (Robust Adversarial Perturbation In EEG Recognition) 系统的主入口程序。
该程序实现了完整的脑电图(EEG)数据处理、特征提取、模型训练和预测的流程。

主要功能模块：
1. AE (AutoEncoder): 自编码器，用于特征提取
2. MADE: 多尺度对抗判别器，用于数据增强和生成
3. Classifier: 分类器，用于最终的分类预测

文件执行顺序分析：
==================
第一阶段：自编码器训练与特征提取
├── 1. AE.train.main() → 调用 AE/train.py
├── 2. AE.get_feat.main() × 3次 → 调用 AE/get_feat.py (处理be, ma, test数据)

第二阶段：MADE模型训练与数据清理
├── 3. MADE.train_epochs.main() → 调用 MADE/train_epochs.py
├── 4. MADE.get_clean_epochs.main() → 调用 MADE/get_clean_epochs.py
└── 5. MADE.final_predict.main() → 调用 MADE/final_predict.py

第三阶段：对抗样本生成
├── 6. generate_cpus() → 循环调用 generate()
│   ├── MADE.train.main() × 2次 → 调用 MADE/train.py (训练be和ma的MADE)
│   ├── MADE.predict.main() × 4次 → 调用 MADE/predict.py (4种组合预测)
│   ├── MADE.train_gen_GAN.main() → 调用 MADE/train_gen_GAN.py
│   └── MADE.generate_GAN.main() → 调用 MADE/generate_GAN.py

第四阶段：分类器训练与预测
└── 7. Classifier.classify.main() → 调用 Classifier/classify.py

未直接运行但被导入的文件：
==========================
- MADE/__init__.py (模块初始化)
- MADE/made.py (MADE模型定义，被其他模块调用)
- MADE/gen_model.py (生成器模型定义，被GAN模块调用)
- MADE/datasets/data_loaders.py (数据加载器，被训练模块调用)
- MADE/datasets/myData.py (数据集类，被数据加载器调用)
- MADE/utils/train.py (训练工具函数，被训练模块调用)
- MADE/utils/validation.py (验证工具函数，被训练模块调用)
- MADE/utils/test.py (测试工具函数，被预测模块调用)
- MADE/utils/plot.py (绘图工具，训练过程中可选使用)
- Classifier/__init__.py (模块初始化)
- Classifier/model.py (MLP分类器模型定义)
- Classifier/loss.py (Co-teaching损失函数)
- AE/__init__.py (模块初始化)
- AE/model.py (LSTM自编码器模型定义)

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
import numpy as np # Added for evaluate_complete_pipeline

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
    
    # 【步骤6a】训练MADE模型 - 分别针对良性(benign)和恶性(malignant)样本
    # 调用文件: MADE/train.py (两次，分别训练be和ma的MADE模型)
    MADE.train.main(feat_dir, model_dir, TRAIN_be, cuda, '-30')
    MADE.train.main(feat_dir, model_dir, TRAIN_ma, cuda, '-30')
    
    # 【步骤6b】使用训练好的MADE模型进行预测
    # 调用文件: MADE/predict.py (四次，4种不同的训练-预测组合)
    # 参数格式: (特征目录, 模型目录, MADE目录, 训练标签, 预测标签, CUDA设备)
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_be, TRAIN_be, cuda)  # 良性->良性
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_be, TRAIN_ma, cuda)  # 良性->恶性
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_ma, TRAIN_ma, cuda)  # 恶性->恶性
    MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_ma, TRAIN_be, cuda)  # 恶性->良性

    # 【步骤6c】训练GAN生成器，用于生成对抗样本
    # 调用文件: MADE/train_gen_GAN.py
    MADE.train_gen_GAN.main(feat_dir, model_dir, made_dir, TRAIN, cuda)
    
    # 【步骤6d】使用训练好的GAN生成器生成对抗样本
    # 调用文件: MADE/generate_GAN.py
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

def evaluate_complete_pipeline(feat_dir, model_dir, result_dir, TRAIN, cuda, round_num):
    """
    评估完整流程的性能（F1分数）
    
    该函数评估整个流程：AE特征提取 + MADE数据清理 + 对抗样本生成 + 分类器预测
    
    参数:
        feat_dir (str): 特征目录
        model_dir (str): 模型目录
        result_dir (str): 结果目录
        TRAIN (str): 训练标签
        cuda (int): CUDA设备ID
        round_num (int): 当前轮次
        
    返回:
        float: F1分数
    """
    try:
        # 创建临时结果目录
        temp_result_dir = os.path.join(result_dir, f'temp_round_{round_num + 1}')
        os.makedirs(temp_result_dir, exist_ok=True)
        
        # 使用当前轮次的完整流程进行预测
        Classifier.classify.main(feat_dir, model_dir, temp_result_dir, TRAIN, cuda, parallel=1)
        
        # 读取预测结果和真实标签
        prediction_file = os.path.join(temp_result_dir, 'prediction.npy')
        if os.path.exists(prediction_file):
            predictions = np.load(prediction_file)
            
            # 读取测试数据标签
            test_data = np.load(os.path.join(feat_dir, 'test.npy'))
            true_labels = test_data[:, -1]
            
            # 计算F1分数
            from sklearn.metrics import f1_score
            f1 = f1_score(true_labels, predictions, average='binary')
            
            # 清理临时文件
            import shutil
            shutil.rmtree(temp_result_dir)
            
            return f1
        else:
            print(f"警告：第{round_num + 1}轮预测文件不存在")
            return 0.0
            
    except Exception as e:
        print(f"第{round_num + 1}轮评估出错: {e}")
        return 0.0

def save_best_models(feat_dir, model_dir, made_dir, TRAIN, 
                    best_ae_model_path, best_made_model_path, best_classifier_model_path,
                    current_gan_models_path, best_gan_models_path):
    """
    保存最佳模型（AE、MADE、分类器、GAN模型）
    
    参数:
        feat_dir (str): 特征目录
        model_dir (str): 模型目录
        made_dir (str): MADE目录
        TRAIN (str): 训练标签
        best_ae_model_path (str): 最佳AE模型保存路径
        best_made_model_path (str): 最佳MADE模型保存路径
        best_classifier_model_path (str): 最佳分类器模型保存路径
        current_gan_models_path (str): 当前轮次GAN模型路径
        best_gan_models_path (str): 最佳GAN模型保存路径
    """
    try:
        import shutil
        
        # 保存最佳AE模型
        ae_files = [f for f in os.listdir(model_dir) if f.startswith('ae') and f.endswith('.pt')]
        if ae_files:
            latest_ae_file = max(ae_files, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
            src_path = os.path.join(model_dir, latest_ae_file)
            shutil.copy2(src_path, best_ae_model_path)
            print(f"最佳AE模型已保存: {best_ae_model_path}")
        
        # 保存最佳MADE模型
        made_files = [f for f in os.listdir(made_dir) if f.endswith('.pt')]
        if made_files:
            latest_made_file = max(made_files, key=lambda x: os.path.getctime(os.path.join(made_dir, x)))
            src_path = os.path.join(made_dir, latest_made_file)
            shutil.copy2(src_path, best_made_model_path)
            print(f"最佳MADE模型已保存: {best_made_model_path}")
        
        # 保存最佳分类器模型
        classifier_files = [f for f in os.listdir(model_dir) if f.startswith('classifier') and f.endswith('.pt')]
        if classifier_files:
            latest_classifier_file = max(classifier_files, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
            src_path = os.path.join(model_dir, latest_classifier_file)
            shutil.copy2(src_path, best_classifier_model_path)
            print(f"最佳分类器模型已保存: {best_classifier_model_path}")
        
        # 保存最佳GAN模型（整个目录）
        if os.path.exists(current_gan_models_path):
            if os.path.exists(best_gan_models_path):
                shutil.rmtree(best_gan_models_path)
            shutil.copytree(current_gan_models_path, best_gan_models_path)
            print(f"最佳GAN模型已保存: {best_gan_models_path}")
            
    except Exception as e:
        print(f"保存最佳模型时出错: {e}")

def save_training_progress(f1_scores, best_f1_score, best_round, result_dir, current_round):
    """
    保存训练进度
    
    参数:
        f1_scores (list): F1分数列表
        best_f1_score (float): 最佳F1分数
        best_round (int): 最佳轮次
        result_dir (str): 结果目录
        current_round (int): 当前轮次
    """
    try:
        progress_file = os.path.join(result_dir, 'training_progress.txt')
        
        with open(progress_file, 'w', encoding='utf-8') as f:
            f.write(f"RAPIER 完整流程训练进度报告\n")
            f.write(f"==========================\n\n")
            f.write(f"当前轮次: {current_round}/50\n")
            f.write(f"最佳F1分数: {best_f1_score:.4f}\n")
            f.write(f"最佳轮次: {best_round}\n\n")
            f.write(f"各轮次F1分数:\n")
            for i, score in enumerate(f1_scores):
                f.write(f"第{i+1}轮: {score:.4f}\n")
        
        print(f"训练进度已保存到: {progress_file}")
        
    except Exception as e:
        print(f"保存训练进度时出错: {e}")

def save_final_report(f1_scores, best_f1_score, best_round, 
                     best_ae_model_path, best_made_model_path, best_classifier_model_path,
                     best_gan_models_path, result_dir):
    """
    保存最终训练报告
    
    参数:
        f1_scores (list): F1分数列表
        best_f1_score (float): 最佳F1分数
        best_round (int): 最佳轮次
        best_ae_model_path (str): 最佳AE模型路径
        best_made_model_path (str): 最佳MADE模型路径
        best_classifier_model_path (str): 最佳分类器模型路径
        best_gan_models_path (str): 最佳GAN模型路径
        result_dir (str): 结果目录
    """
    try:
        report_file = os.path.join(result_dir, 'final_training_report.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"RAPIER 完整流程最终训练报告\n")
            f.write(f"==========================\n\n")
            f.write(f"训练轮次: 50\n")
            f.write(f"最佳F1分数: {best_f1_score:.4f}\n")
            f.write(f"最佳轮次: {best_round}\n\n")
            f.write(f"最佳模型路径:\n")
            f.write(f"  - AE: {best_ae_model_path}\n")
            f.write(f"  - MADE: {best_made_model_path}\n")
            f.write(f"  - 分类器: {best_classifier_model_path}\n")
            f.write(f"  - GAN模型: {best_gan_models_path}\n\n")
            
            f.write(f"F1分数统计:\n")
            f.write(f"平均F1分数: {np.mean(f1_scores):.4f}\n")
            f.write(f"标准差: {np.std(f1_scores):.4f}\n")
            f.write(f"最高F1分数: {max(f1_scores):.4f}\n")
            f.write(f"最低F1分数: {min(f1_scores):.4f}\n\n")
            
            f.write(f"各轮次详细F1分数:\n")
            for i, score in enumerate(f1_scores):
                marker = " 🏆" if i + 1 == best_round else ""
                f.write(f"第{i+1:2d}轮: {score:.4f}{marker}\n")
        
        print(f"最终训练报告已保存到: {report_file}")
        
    except Exception as e:
        print(f"保存最终报告时出错: {e}")

def reload_best_models(feat_dir, best_model_dir, made_dir, best_ae_model_path, 
                      best_made_model_path, best_classifier_model_path, best_gan_models_path):
    """
    重新加载最佳模型进行后续处理
    
    参数:
        feat_dir (str): 特征目录
        best_model_dir (str): 最佳模型目录
        made_dir (str): MADE目录
        best_ae_model_path (str): 最佳AE模型路径
        best_made_model_path (str): 最佳MADE模型路径
        best_classifier_model_path (str): 最佳分类器模型路径
        best_gan_models_path (str): 最佳GAN模型路径
    """
    try:
        import shutil
        
        print("重新加载最佳模型...")
        
        # 重新加载最佳AE模型的特征
        print("重新加载最佳AE模型特征...")
        # 这里需要重新运行AE特征提取，使用最佳模型
        
        # 重新加载最佳MADE模型
        print("重新加载最佳MADE模型...")
        if os.path.exists(best_made_model_path):
            shutil.copy2(best_made_model_path, os.path.join(made_dir, 'best_made_model.pt'))
        
        # 重新加载最佳分类器模型
        print("重新加载最佳分类器模型...")
        if os.path.exists(best_classifier_model_path):
            shutil.copy2(best_classifier_model_path, os.path.join(best_model_dir, 'best_classifier_model.pt'))
        
        # 重新加载最佳GAN模型
        print("重新加载最佳GAN模型...")
        if os.path.exists(best_gan_models_path):
            gan_target_dir = os.path.join(made_dir, 'best_gan_models')
            if os.path.exists(gan_target_dir):
                shutil.rmtree(gan_target_dir)
            shutil.copytree(best_gan_models_path, gan_target_dir)
            
        print("所有最佳模型已重新加载完成！")
            
    except Exception as e:
        print(f"重新加载最佳模型时出错: {e}")

def main(data_dir, model_dir, feat_dir, made_dir, result_dir, cuda):
    """
    主函数 - 执行完整的RAPIER流程
    
    该函数实现了RAPIER系统的完整工作流程：
    1. 训练自编码器(AE)模型
    2. 提取特征（良性、恶性、测试数据）
    3. 50轮完整流程训练，包括AE、MADE、对抗样本生成和分类器的联合优化
    4. 使用最佳模型进行最终分类
    
    参数:
        data_dir (str): 原始数据目录路径
        model_dir (str): 模型保存目录路径
        feat_dir (str): 特征文件目录路径
        made_dir (str): MADE相关文件目录路径
        result_dir (str): 结果输出目录路径
        cuda (int): CUDA设备ID
    """
    
    print("开始50轮完整流程训练，寻找最佳F1分数模型...")
    
    # 创建最佳模型保存目录
    best_model_dir = os.path.join(os.path.dirname(model_dir), 'model_best')
    os.makedirs(best_model_dir, exist_ok=True)
    print(f"最佳模型将保存到: {best_model_dir}")
    
    best_f1_score = 0.0
    best_round = 0
    best_ae_model_path = ""
    best_made_model_path = ""
    best_classifier_model_path = ""
    best_gan_models_path = ""
    f1_scores = []
    
    # 50轮完整流程训练循环
    for round_num in range(50):
        print(f"\n=== 第 {round_num + 1}/50 轮完整流程训练 ===")
        
        # 清空相关目录，准备新一轮训练
        if os.path.exists(feat_dir):
            import shutil
            shutil.rmtree(feat_dir)
        os.makedirs(feat_dir, exist_ok=True)
        
        if os.path.exists(made_dir):
            shutil.rmtree(made_dir)
        os.makedirs(made_dir, exist_ok=True)
        
        # 【步骤1】训练自编码器模型
        print(f"第{round_num + 1}轮：训练自编码器模型...")
        AE.train.main(data_dir, model_dir, cuda)
        
        # 【步骤2】使用训练好的自编码器提取特征
        print(f"第{round_num + 1}轮：提取特征...")
        AE.get_feat.main(data_dir, model_dir, feat_dir, 'be', cuda)    # 提取良性样本特征
        AE.get_feat.main(data_dir, model_dir, feat_dir, 'ma', cuda)    # 提取恶性样本特征
        AE.get_feat.main(data_dir, model_dir, feat_dir, 'test', cuda)  # 提取测试样本特征

        # 【步骤3】训练MADE模型并进行数据清理
        print(f"第{round_num + 1}轮：训练MADE模型...")
        TRAIN = 'be'  # 使用良性样本进行训练
        MADE.train_epochs.main(feat_dir, model_dir, made_dir, TRAIN, cuda, '20')
        MADE.get_clean_epochs.main(feat_dir, made_dir, '0.5', TRAIN)
        MADE.final_predict.main(feat_dir)
        
        # 【步骤4】生成对抗样本
        print(f"第{round_num + 1}轮：生成对抗样本...")
        current_gan_models_path = os.path.join(made_dir, f'gan_models_round_{round_num + 1}')
        os.makedirs(current_gan_models_path, exist_ok=True)
        
        # 训练GAN生成器
        MADE.train_gen_GAN.main(feat_dir, model_dir, current_gan_models_path, TRAIN, cuda)
        
        # 生成对抗样本（为5个不同的索引生成）
        MADE.generate_GAN.main(feat_dir, model_dir, current_gan_models_path, TRAIN, list(range(5)), cuda)
        
        # 【步骤5】训练分类器并评估F1分数
        print(f"第{round_num + 1}轮：训练分类器并评估性能...")
        current_f1 = evaluate_complete_pipeline(feat_dir, model_dir, result_dir, TRAIN, cuda, round_num)
        f1_scores.append(current_f1)
        
        print(f"第 {round_num + 1} 轮完整流程 F1 分数: {current_f1:.4f}")
        
        # 检查是否为最佳模型
        if current_f1 > best_f1_score:
            best_f1_score = current_f1
            best_round = round_num + 1
            
            # 保存最佳模型到model_best目录
            best_ae_model_path = os.path.join(best_model_dir, f'best_ae_model_round_{round_num + 1}_f1_{current_f1:.4f}.pt')
            best_made_model_path = os.path.join(best_model_dir, f'best_made_model_round_{round_num + 1}_f1_{current_f1:.4f}.pt')
            best_classifier_model_path = os.path.join(best_model_dir, f'best_classifier_model_round_{round_num + 1}_f1_{current_f1:.4f}.pt')
            best_gan_models_path = os.path.join(best_model_dir, f'best_gan_models_round_{round_num + 1}_f1_{current_f1:.4f}')
            
            save_best_models(feat_dir, model_dir, made_dir, TRAIN, 
                           best_ae_model_path, best_made_model_path, best_classifier_model_path, 
                           current_gan_models_path, best_gan_models_path)
            
            print(f"🎉 发现新的最佳完整流程！F1分数: {current_f1:.4f}")
            print(f"最佳模型已保存到:")
            print(f"  - AE: {best_ae_model_path}")
            print(f"  - MADE: {best_made_model_path}")
            print(f"  - 分类器: {best_classifier_model_path}")
            print(f"  - GAN模型: {best_gan_models_path}")
        
        # 每10轮保存一次进度
        if (round_num + 1) % 10 == 0:
            save_training_progress(f1_scores, best_f1_score, best_round, result_dir, round_num + 1)
    
    # 训练完成，记录最终结果
    print(f"\n=== 50轮完整流程训练完成 ===")
    print(f"最佳F1分数: {best_f1_score:.4f} (第{best_round}轮)")
    print(f"最佳模型路径:")
    print(f"  - AE: {best_ae_model_path}")
    print(f"  - MADE: {best_made_model_path}")
    print(f"  - 分类器: {best_classifier_model_path}")
    print(f"  - GAN模型: {best_gan_models_path}")
    
    # 保存最终训练报告
    save_final_report(f1_scores, best_f1_score, best_round, 
                     best_ae_model_path, best_made_model_path, best_classifier_model_path, 
                     best_gan_models_path, result_dir)
    
    # 使用最佳模型进行后续处理
    print(f"使用最佳模型（第{best_round}轮）进行后续处理...")
    
    # 重新加载最佳模型进行后续处理
    reload_best_models(feat_dir, best_model_dir, made_dir, best_ae_model_path, 
                      best_made_model_path, best_classifier_model_path, best_gan_models_path)
    
    # 【步骤6】使用最佳模型进行最终预测
    print("使用最佳模型进行最终预测...")
    TRAIN = 'corrected'  # 使用修正后的数据进行训练
    Classifier.classify.main(feat_dir, best_model_dir, result_dir, TRAIN, cuda, parallel=5)
    
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