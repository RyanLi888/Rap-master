#!/usr/bin/env python3
"""
RAPIER 可重复性测试脚本
=====================

本脚本用于验证RAPIER系统的可重复性修复是否有效。
通过多次运行相同的配置，检查F1-score的稳定性。

使用方法:
    python test_reproducibility.py

作者: RAPIER 开发团队
版本: 1.0
"""

import os
import sys
import numpy as np
import time
from datetime import datetime

# 添加主程序路径
sys.path.append('main')
sys.path.append('utils')

try:
    from main import main as rapier_main
    from random_seed import set_random_seed, RANDOM_CONFIG
    print("✅ RAPIER主程序和随机种子控制模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)


def run_single_experiment(run_id, random_seed=42):
    """
    运行单次RAPIER实验
    
    参数:
        run_id (int): 运行编号
        random_seed (int): 随机种子
        
    返回:
        dict: 包含运行结果的字典
    """
    print(f"\n{'='*50}")
    print(f"🧪 开始第 {run_id} 次实验")
    print(f"🎯 使用随机种子: {random_seed}")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")
    
    # 设置目录路径
    data_dir = 'data/data'
    feat_dir = 'data/feat'
    model_dir = 'data/model'
    made_dir = 'data/made'
    result_dir = f'data/result_test_{run_id}'
    cuda = 0
    
    # 创建结果目录
    os.makedirs(result_dir, exist_ok=True)
    
    try:
        start_time = time.time()
        
        # 运行RAPIER主程序
        rapier_main(data_dir, model_dir, feat_dir, made_dir, result_dir, cuda, random_seed)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 读取结果
        prediction_file = os.path.join(result_dir, 'prediction.npy')
        if os.path.exists(prediction_file):
            predictions = np.load(prediction_file)
            
            # 读取测试标签
            test_data = np.load(os.path.join(feat_dir, 'test.npy'))
            true_labels = test_data[:, -1]
            
            # 计算评估指标
            from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
            f1 = f1_score(true_labels, predictions, average='binary')
            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions, average='binary')
            recall = recall_score(true_labels, predictions, average='binary')
            
            result = {
                'run_id': run_id,
                'random_seed': random_seed,
                'f1_score': f1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'duration': duration,
                'status': 'success',
                'error': None
            }
            
            print(f"✅ 第 {run_id} 次实验完成")
            print(f"📊 F1分数: {f1:.6f}")
            print(f"📊 准确率: {accuracy:.6f}")
            print(f"📊 精确率: {precision:.6f}")
            print(f"📊 召回率: {recall:.6f}")
            print(f"⏱️  耗时: {duration:.2f}秒")
            
        else:
            result = {
                'run_id': run_id,
                'random_seed': random_seed,
                'f1_score': 0.0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'duration': duration,
                'status': 'failed',
                'error': 'prediction file not found'
            }
            print(f"❌ 第 {run_id} 次实验失败: 预测文件未找到")
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        result = {
            'run_id': run_id,
            'random_seed': random_seed,
            'f1_score': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'duration': duration,
            'status': 'error',
            'error': str(e)
        }
        print(f"❌ 第 {run_id} 次实验出错: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def analyze_results(results):
    """
    分析实验结果的稳定性
    
    参数:
        results (list): 实验结果列表
    """
    print(f"\n{'='*60}")
    print("📈 实验结果分析")
    print(f"{'='*60}")
    
    successful_results = [r for r in results if r['status'] == 'success']
    
    if len(successful_results) == 0:
        print("❌ 没有成功的实验结果")
        return
    
    # 提取指标
    f1_scores = [r['f1_score'] for r in successful_results]
    accuracies = [r['accuracy'] for r in successful_results]
    precisions = [r['precision'] for r in successful_results]
    recalls = [r['recall'] for r in successful_results]
    durations = [r['duration'] for r in successful_results]
    
    # 统计分析
    def analyze_metric(values, name):
        if len(values) == 0:
            return
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        range_val = max_val - min_val
        cv = std_val / mean_val * 100 if mean_val != 0 else 0  # 变异系数
        
        print(f"\n📊 {name}:")
        print(f"  平均值: {mean_val:.6f}")
        print(f"  标准差: {std_val:.6f}")
        print(f"  最小值: {min_val:.6f}")
        print(f"  最大值: {max_val:.6f}")
        print(f"  范围:   {range_val:.6f}")
        print(f"  变异系数: {cv:.2f}%")
        
        # 可重复性判断
        if name == "F1分数":
            if std_val < 0.001:
                print("  ✅ 极高稳定性 (标准差 < 0.001)")
            elif std_val < 0.01:
                print("  ✅ 高稳定性 (标准差 < 0.01)")
            elif std_val < 0.05:
                print("  ⚠️  中等稳定性 (标准差 < 0.05)")
            else:
                print("  ❌ 低稳定性 (标准差 >= 0.05)")
    
    analyze_metric(f1_scores, "F1分数")
    analyze_metric(accuracies, "准确率")
    analyze_metric(precisions, "精确率")
    analyze_metric(recalls, "召回率")
    analyze_metric(durations, "运行时间(秒)")
    
    # 详细结果表格
    print(f"\n📋 详细结果:")
    print(f"{'运行次数':<8} {'F1分数':<10} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'耗时(秒)':<10}")
    print("-" * 70)
    for r in successful_results:
        print(f"{r['run_id']:<8} {r['f1_score']:<10.6f} {r['accuracy']:<10.6f} "
              f"{r['precision']:<10.6f} {r['recall']:<10.6f} {r['duration']:<10.2f}")
    
    # 失败分析
    failed_results = [r for r in results if r['status'] != 'success']
    if len(failed_results) > 0:
        print(f"\n❌ 失败的实验 ({len(failed_results)} 次):")
        for r in failed_results:
            print(f"  运行 {r['run_id']}: {r['error']}")


def main():
    """主函数"""
    print("🚀 RAPIER 可重复性测试开始")
    print(f"📅 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 测试参数
    num_runs = 3  # 运行次数（建议3-5次）
    fixed_seed = 42  # 固定种子
    
    print(f"🔧 测试配置:")
    print(f"  运行次数: {num_runs}")
    print(f"  固定种子: {fixed_seed}")
    print(f"  预期结果: 所有运行应产生相同的F1分数")
    
    # 运行多次实验
    results = []
    for i in range(1, num_runs + 1):
        result = run_single_experiment(i, fixed_seed)
        results.append(result)
        
        # 避免连续运行时的资源冲突
        if i < num_runs:
            print(f"⏳ 等待 5 秒后开始下一次实验...")
            time.sleep(5)
    
    # 分析结果
    analyze_results(results)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"reproducibility_test_results_{timestamp}.txt"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"RAPIER 可重复性测试结果\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"运行次数: {num_runs}\n")
        f.write(f"固定种子: {fixed_seed}\n\n")
        
        f.write("详细结果:\n")
        f.write(f"{'运行次数':<8} {'F1分数':<12} {'准确率':<12} {'精确率':<12} {'召回率':<12} {'状态':<10}\n")
        f.write("-" * 80 + "\n")
        
        for r in results:
            f.write(f"{r['run_id']:<8} {r['f1_score']:<12.6f} {r['accuracy']:<12.6f} "
                   f"{r['precision']:<12.6f} {r['recall']:<12.6f} {r['status']:<10}\n")
    
    print(f"\n💾 测试结果已保存到: {result_file}")
    print("🎯 可重复性测试完成!")
    
    # 给出建议
    successful_results = [r for r in results if r['status'] == 'success']
    if len(successful_results) >= 2:
        f1_scores = [r['f1_score'] for r in successful_results]
        f1_std = np.std(f1_scores)
        
        if f1_std < 0.001:
            print("✅ 修复成功！F1分数具有极高的可重复性")
        elif f1_std < 0.01:
            print("✅ 修复成功！F1分数具有良好的可重复性") 
        else:
            print("⚠️  仍存在一定的随机性，可能需要进一步调优")


if __name__ == "__main__":
    main()
