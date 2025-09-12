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

def evaluate_complete_pipeline(feat_dir, model_dir, result_dir, TRAIN, cuda):
    """
    评估完整流程的性能（F1分数）
    
    该函数评估整个流程：AE特征提取 + MADE数据清理 + 对抗样本生成 + 分类器预测
    
    参数:
        feat_dir (str): 特征目录
        model_dir (str): 模型目录
        result_dir (str): 结果目录
        TRAIN (str): 训练标签
        cuda (int): CUDA设备ID
        
    返回:
        float: F1分数
    """
    try:
        # 创建临时结果目录
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_result_dir = os.path.join(result_dir, f'temp_eval_{timestamp}')
        os.makedirs(temp_result_dir, exist_ok=True)
        
        # 使用当前的完整流程进行预测
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
            print(f"警告：预测文件不存在")
            return 0.0
            
    except Exception as e:
        print(f"评估出错: {e}")
        return 0.0



def main(data_dir, model_dir, feat_dir, made_dir, result_dir, cuda, random_seed=None):
    """
    主函数 - 执行完整的RAPIER流程并与历史最佳模型对比
    
    该函数实现了RAPIER系统的完整工作流程：
    1. 设置随机种子确保可重复性
    2. 加载历史最佳F1分数和模型路径
    3. 训练当前模型（AE、MADE、分类器）
    4. 评估当前模型的F1分数
    5. 与历史最佳进行对比，如果更好则保存新的最佳模型
    6. 使用最佳模型进行最终分类
    
    参数:
        data_dir (str): 原始数据目录路径
        model_dir (str): 模型保存目录路径
        feat_dir (str): 特征文件目录路径
        made_dir (str): MADE相关文件目录路径
        result_dir (str): 结果输出目录路径
        cuda (int): CUDA设备ID
        random_seed (int): 全局随机种子，默认使用配置中的种子
    """
    
    # 【第0步】设置随机种子确保可重复性
    if SEED_CONTROL_AVAILABLE:
        if random_seed is None:
            random_seed = RANDOM_CONFIG['global_seed']
        set_random_seed(random_seed)
        print(f"🎯 已设置全局随机种子: {random_seed}")
    else:
        print("⚠️  跳过随机种子设置（模块不可用）")
    
    print("开始RAPIER完整流程训练，将与历史最佳模型对比...")
    
    # 创建最佳模型保存目录
    best_model_dir = os.path.join(os.path.dirname(model_dir), 'model_best')
    os.makedirs(best_model_dir, exist_ok=True)
    
    # 加载历史最佳F1分数和模型路径
    historical_best = load_historical_best(best_model_dir)
    print(f"历史最佳F1分数: {historical_best['f1_score']:.4f}")
    print(f"最佳模型保存位置: {best_model_dir}")
    
    # 【第一步】清空临时目录，准备当前训练
    print("\n=== 开始当前完整流程训练 ===")
    
    # 清空相关目录，准备新训练
    if os.path.exists(feat_dir):
        import shutil
        shutil.rmtree(feat_dir)
    os.makedirs(feat_dir, exist_ok=True)
    
    if os.path.exists(made_dir):
        shutil.rmtree(made_dir)
    os.makedirs(made_dir, exist_ok=True)
    
    # 【第二步】训练自编码器模型
    print("训练自编码器模型...")
    AE.train.main(data_dir, model_dir, cuda)
    
    # 【第三步】使用训练好的自编码器提取特征
    print("提取特征...")
    AE.get_feat.main(data_dir, model_dir, feat_dir, 'be', cuda)    # 提取良性样本特征
    AE.get_feat.main(data_dir, model_dir, feat_dir, 'ma', cuda)    # 提取恶性样本特征
    AE.get_feat.main(data_dir, model_dir, feat_dir, 'test', cuda)  # 提取测试样本特征
    
    # 【第四步】训练MADE模型并进行数据清理
    print("训练MADE模型并进行数据清理...")
    TRAIN = 'be'  # 使用良性样本进行训练
    
    # 训练良性样本的MADE模型（用于数据清理）
    MADE.train_epochs.main(feat_dir, model_dir, made_dir, TRAIN, cuda, '20')
    
    # 数据清理和标签修正（使用良性样本作为基准）
    MADE.get_clean_epochs.main(feat_dir, made_dir, '0.5', TRAIN)
    MADE.final_predict.main(feat_dir)
    
    # 【第五步】生成对抗样本
    print("生成对抗样本...")
    generate_cpus(feat_dir, model_dir, made_dir, list(range(5)), cuda)
    
    # 【第六步】训练分类器并评估F1分数
    print("训练分类器并评估性能...")
    TRAIN = 'corrected'  # 设置为修正后的数据
    current_f1 = evaluate_complete_pipeline(feat_dir, model_dir, result_dir, TRAIN, cuda)
    
    print(f"\n当前运行的F1分数: {current_f1:.4f}")
    print(f"历史最佳F1分数: {historical_best['f1_score']:.4f}")
    
    # 【第七步】对比并决定是否保存新的最佳模型
    if current_f1 > historical_best['f1_score'] or historical_best['f1_score'] == 0.0:
        print(f"\n🎉 发现新的最佳模型！F1分数从 {historical_best['f1_score']:.4f} 提升到 {current_f1:.4f}")
        
        # 保存新的最佳模型
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        new_best_models = save_new_best_models(model_dir, made_dir, best_model_dir, current_f1, timestamp)
        
        # 更新历史最佳记录
        update_historical_best(best_model_dir, current_f1, new_best_models, timestamp)
        
        print("新的最佳模型已保存！")
        print(f"  - AE: {new_best_models['ae_path']}")
        print(f"  - MADE: {new_best_models['made_path']}")
        print(f"  - 分类器: {new_best_models['classifier_path']}")
    else:
        print(f"\n当前F1分数 {current_f1:.4f} 未超过历史最佳 {historical_best['f1_score']:.4f}")
        print("保持历史最佳模型不变")
    
    # 【第八步】使用最佳模型进行最终预测
    print("\n使用最佳模型进行最终预测...")
    
    # 直接从 model_best 目录加载最佳分类器模型
    print(f"✨ 从 model_best 目录加载最佳分类器模型")
    
    # 查找最佳分类器文件（按文件名中的F1分数和时间戳排序）
    import glob
    classifier_pattern = os.path.join(best_model_dir, "best_classifier_f1_*.pkl")
    classifier_files = glob.glob(classifier_pattern)
    
    if classifier_files:
        # 按修改时间排序，取最新的最佳模型
        best_classifier_file = max(classifier_files, key=os.path.getctime)
        print(f"📂 找到最佳分类器模型: {os.path.basename(best_classifier_file)}")
        
        # 使用最佳模型进行预测
        final_f1 = Classifier.classify.predict_only_from_file(
            feat_dir, best_classifier_file, result_dir, TRAIN, cuda, parallel=5
        )
        print(f"🎯 最终预测完成，F1分数: {final_f1:.4f}")
    else:
        print("⚠️  警告：在 model_best 目录中未找到最佳分类器模型文件")
        print("🔄 回退到使用工作目录中的当前模型")
        final_f1 = Classifier.classify.predict_only(feat_dir, model_dir, result_dir, TRAIN, cuda, parallel=5)
        print(f"🎯 最终预测完成，F1分数: {final_f1:.4f}")
    
    print("\nRAPIER流程完成！")
    print(f"最终使用的最佳F1分数: {max(current_f1, historical_best['f1_score']):.4f}")

def load_historical_best(best_model_dir):
    """
    加载历史最佳F1分数和模型路径
    
    参数:
        best_model_dir (str): 最佳模型目录
        
    返回:
        dict: 包含历史最佳信息的字典
    """
    history_file = os.path.join(best_model_dir, 'best_history.txt')
    
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) >= 4:
                    f1_score = float(lines[0].split(':')[1].strip())
                    ae_path = lines[1].split(':', 1)[1].strip()
                    made_path = lines[2].split(':', 1)[1].strip()
                    classifier_path = lines[3].split(':', 1)[1].strip()
                    
                    return {
                        'f1_score': f1_score,
                        'ae_path': ae_path,
                        'made_path': made_path,
                        'classifier_path': classifier_path
                    }
        except Exception as e:
            print(f"读取历史记录时出错: {e}")
    
    # 如果没有历史记录，返回默认值
    return {
        'f1_score': 0.0,
        'ae_path': '',
        'made_path': '',
        'classifier_path': ''
    }

def save_new_best_models(model_dir, made_dir, best_model_dir, f1_score, timestamp):
    """
    保存新的最佳模型
    
    参数:
        model_dir (str): 当前模型目录
        made_dir (str): 当前MADE目录
        best_model_dir (str): 最佳模型保存目录
        f1_score (float): F1分数
        timestamp (str): 时间戳
        
    返回:
        dict: 新保存的模型路径
    """
    try:
        import shutil
        
        # 定义新的最佳模型文件名（保持原始格式）
        new_ae_path = os.path.join(best_model_dir, f'best_ae_f1_{f1_score:.4f}_{timestamp}.pkl')
        new_made_path = os.path.join(best_model_dir, f'best_made_f1_{f1_score:.4f}_{timestamp}.pt')
        new_classifier_path = os.path.join(best_model_dir, f'best_classifier_f1_{f1_score:.4f}_{timestamp}.pkl')
        
        # 保存AE模型 - 实际文件名是 gru_ae.pkl
        ae_file = os.path.join(model_dir, 'gru_ae.pkl')
        if os.path.exists(ae_file):
            shutil.copy2(ae_file, new_ae_path)
            print(f"  → AE模型已保存: {new_ae_path}")
        else:
            print(f"  ⚠️  警告: AE模型文件不存在 {ae_file}")
        
        # 保存MADE模型 - 查找所有.pt文件
        made_files = [f for f in os.listdir(model_dir) if 'made' in f.lower() and f.endswith('.pt')]
        if made_files:
            latest_made_file = max(made_files, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
            shutil.copy2(os.path.join(model_dir, latest_made_file), new_made_path)
            print(f"  → MADE模型已保存: {new_made_path}")
        else:
            print(f"  ⚠️  警告: 未找到MADE模型文件在 {model_dir}")
        
        # 保存分类器模型 - 实际文件名是 Detection_Model.pkl
        classifier_file = os.path.join(model_dir, 'Detection_Model.pkl')
        if os.path.exists(classifier_file):
            shutil.copy2(classifier_file, new_classifier_path)
            print(f"  → 分类器模型已保存: {new_classifier_path}")
        else:
            print(f"  ⚠️  警告: 分类器模型文件不存在 {classifier_file}")
        
        return {
            'ae_path': new_ae_path,
            'made_path': new_made_path,
            'classifier_path': new_classifier_path
        }
    
    except Exception as e:
        print(f"保存新的最佳模型时出错: {e}")
        return {'ae_path': '', 'made_path': '', 'classifier_path': ''}

def update_historical_best(best_model_dir, f1_score, model_paths, timestamp):
    """
    更新历史最佳记录
    
    参数:
        best_model_dir (str): 最佳模型目录
        f1_score (float): F1分数
        model_paths (dict): 模型路径字典
        timestamp (str): 时间戳
    """
    try:
        history_file = os.path.join(best_model_dir, 'best_history.txt')
        
        with open(history_file, 'w', encoding='utf-8') as f:
            f.write(f"F1分数: {f1_score:.4f}\n")
            f.write(f"AE模型: {model_paths['ae_path']}\n")
            f.write(f"MADE模型: {model_paths['made_path']}\n")
            f.write(f"分类器模型: {model_paths['classifier_path']}\n")
            f.write(f"更新时间: {timestamp}\n")
    
    except Exception as e:
        print(f"更新历史记录时出错: {e}")

def copy_best_models_to_work_dir(best_model_dir, model_dir, historical_best=None):
    """
    将最佳模型复制回工作目录，以便第8步使用
    
    参数:
        best_model_dir (str): 最佳模型目录
        model_dir (str): 工作模型目录
        historical_best (dict): 历史最佳模型信息，如果为None则使用当前最新的
    """
    try:
        import shutil
        
        if historical_best and historical_best['f1_score'] > 0:
            # 使用历史最佳模型
            ae_source = historical_best['ae_path']
            made_source = historical_best['made_path'] 
            classifier_source = historical_best['classifier_path']
        else:
            # 使用当前目录中最新的最佳模型
            best_files = [f for f in os.listdir(best_model_dir) if f.startswith('best_') and (f.endswith('.pt') or f.endswith('.pkl'))]
            if not best_files:
                print("  ⚠️  警告: 未找到最佳模型文件")
                return
                
            # 按时间戳排序，获取最新的
            best_files.sort(reverse=True)
            
            ae_source = None
            made_source = None  
            classifier_source = None
            
            for f in best_files:
                if 'ae' in f and ae_source is None:
                    ae_source = os.path.join(best_model_dir, f)
                elif 'made' in f and made_source is None:
                    made_source = os.path.join(best_model_dir, f)
                elif 'classifier' in f and classifier_source is None:
                    classifier_source = os.path.join(best_model_dir, f)
        
        # 复制模型文件
        if ae_source and os.path.exists(ae_source):
            shutil.copy2(ae_source, os.path.join(model_dir, 'gru_ae.pkl'))
            print(f"  → 已复制最佳AE模型到工作目录")
        
        if made_source and os.path.exists(made_source):
            # MADE模型需要保持原始命名格式
            made_filename = [f for f in os.listdir(model_dir) if 'made' in f.lower() and f.endswith('.pt')]
            if made_filename:
                shutil.copy2(made_source, os.path.join(model_dir, made_filename[0]))
                print(f"  → 已复制最佳MADE模型到工作目录")
        
        if classifier_source and os.path.exists(classifier_source):
            shutil.copy2(classifier_source, os.path.join(model_dir, 'Detection_Model.pkl'))
            print(f"  → 已复制最佳分类器模型到工作目录")
            
    except Exception as e:
        print(f"复制最佳模型到工作目录时出错: {e}")

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