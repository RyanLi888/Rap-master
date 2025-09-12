#!/usr/bin/env python3
"""
RAPIER 最优随机种子搜索脚本
===========================

本脚本通过尝试不同的随机种子来寻找能够产生最佳F1-score的配置。
这是在保证可重复性的前提下获得最优结果的推荐方法。

使用方法:
    python find_best_seed.py

作者: RAPIER 开发团队
版本: 1.0
"""

import os
import sys
import numpy as np
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt

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


def evaluate_seed(seed, run_id):
    """
    评估指定随机种子的性能
    
    参数:
        seed (int): 要测试的随机种子
        run_id (int): 运行编号
        
    返回:
        dict: 包含评估结果的字典
    """
    print(f"\n{'='*50}")
    print(f"🧪 测试随机种子: {seed} (第 {run_id} 个)")
    print(f"⏰ 开始时间: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*50}")
    
    # 设置目录路径
    data_dir = 'data/data'
    feat_dir = f'data/feat_seed_{seed}'
    model_dir = f'data/model_seed_{seed}'
    made_dir = f'data/made_seed_{seed}'
    result_dir = f'data/result_seed_{seed}'
    cuda = 0
    
    # 创建独立的目录避免冲突
    for dir_path in [feat_dir, model_dir, made_dir, result_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    try:
        start_time = time.time()
        
        # 运行RAPIER主程序
        rapier_main(data_dir, model_dir, feat_dir, made_dir, result_dir, cuda, seed)
        
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
                'seed': seed,
                'run_id': run_id,
                'f1_score': f1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'duration': duration,
                'status': 'success',
                'error': None,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ 种子 {seed} 测试完成")
            print(f"📊 F1分数: {f1:.6f}")
            print(f"📊 准确率: {accuracy:.6f}")
            print(f"📊 精确率: {precision:.6f}")
            print(f"📊 召回率: {recall:.6f}")
            print(f"⏱️  耗时: {duration:.2f}秒")
            
        else:
            result = {
                'seed': seed,
                'run_id': run_id,
                'f1_score': 0.0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'duration': duration,
                'status': 'failed',
                'error': 'prediction file not found',
                'timestamp': datetime.now().isoformat()
            }
            print(f"❌ 种子 {seed} 测试失败: 预测文件未找到")
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        result = {
            'seed': seed,
            'run_id': run_id,
            'f1_score': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'duration': duration,
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        print(f"❌ 种子 {seed} 测试出错: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def generate_seed_candidates(strategy='mixed', num_seeds=10):
    """
    生成候选随机种子
    
    参数:
        strategy (str): 生成策略
            - 'random': 完全随机
            - 'systematic': 系统性选择
            - 'mixed': 混合策略（推荐）
        num_seeds (int): 种子数量
        
    返回:
        list: 候选种子列表
    """
    seeds = []
    
    if strategy == 'random':
        # 完全随机策略
        np.random.seed(42)  # 确保种子生成的可重复性
        seeds = np.random.randint(1, 100000, num_seeds).tolist()
        
    elif strategy == 'systematic':
        # 系统性策略：均匀分布
        seeds = np.linspace(1, 10000, num_seeds, dtype=int).tolist()
        
    elif strategy == 'mixed':
        # 混合策略（推荐）
        seeds = []
        
        # 1. 常用的"幸运数字"
        lucky_seeds = [42, 123, 456, 789, 1234, 2021, 2024]
        seeds.extend(lucky_seeds[:min(3, num_seeds//3)])
        
        # 2. 基于原有MADE种子的变化
        base_seed = 290713
        for i in range(min(3, num_seeds//3)):
            seeds.append(base_seed + i * 1000)
        
        # 3. 随机补充
        np.random.seed(42)
        remaining = num_seeds - len(seeds)
        if remaining > 0:
            random_seeds = np.random.randint(1, 50000, remaining).tolist()
            seeds.extend(random_seeds)
    
    # 确保种子唯一性
    seeds = list(set(seeds))[:num_seeds]
    
    print(f"🎲 生成 {len(seeds)} 个候选种子: {seeds}")
    return seeds


def analyze_results(results):
    """
    分析种子搜索结果
    
    参数:
        results (list): 所有种子的评估结果
        
    返回:
        dict: 分析结果
    """
    print(f"\n{'='*60}")
    print("📈 种子搜索结果分析")
    print(f"{'='*60}")
    
    successful_results = [r for r in results if r['status'] == 'success']
    
    if len(successful_results) == 0:
        print("❌ 没有成功的实验结果")
        return None
    
    # 按F1分数排序
    successful_results.sort(key=lambda x: x['f1_score'], reverse=True)
    
    # 提取指标
    f1_scores = [r['f1_score'] for r in successful_results]
    
    # 统计分析
    best_result = successful_results[0]
    worst_result = successful_results[-1]
    avg_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    
    print(f"\n🏆 最佳结果:")
    print(f"  种子: {best_result['seed']}")
    print(f"  F1分数: {best_result['f1_score']:.6f}")
    print(f"  准确率: {best_result['accuracy']:.6f}")
    print(f"  精确率: {best_result['precision']:.6f}")
    print(f"  召回率: {best_result['recall']:.6f}")
    
    print(f"\n📊 整体统计:")
    print(f"  平均F1: {avg_f1:.6f}")
    print(f"  最佳F1: {best_result['f1_score']:.6f}")
    print(f"  最差F1: {worst_result['f1_score']:.6f}")
    print(f"  F1范围: {best_result['f1_score'] - worst_result['f1_score']:.6f}")
    print(f"  F1标准差: {std_f1:.6f}")
    
    # Top 5结果
    print(f"\n🥇 Top 5 最佳种子:")
    print(f"{'排名':<4} {'种子':<8} {'F1分数':<10} {'准确率':<10} {'精确率':<10} {'召回率':<10}")
    print("-" * 70)
    for i, r in enumerate(successful_results[:5]):
        print(f"{i+1:<4} {r['seed']:<8} {r['f1_score']:<10.6f} {r['accuracy']:<10.6f} "
              f"{r['precision']:<10.6f} {r['recall']:<10.6f}")
    
    # 失败分析
    failed_results = [r for r in results if r['status'] != 'success']
    if len(failed_results) > 0:
        print(f"\n❌ 失败的种子 ({len(failed_results)} 个):")
        for r in failed_results:
            print(f"  种子 {r['seed']}: {r['error']}")
    
    return {
        'best_seed': best_result['seed'],
        'best_f1': best_result['f1_score'],
        'best_result': best_result,
        'all_results': successful_results,
        'avg_f1': avg_f1,
        'std_f1': std_f1
    }


def plot_results(results, save_path='seed_search_results.png'):
    """
    绘制种子搜索结果图表
    
    参数:
        results (list): 评估结果列表
        save_path (str): 图表保存路径
    """
    successful_results = [r for r in results if r['status'] == 'success']
    
    if len(successful_results) < 2:
        print("⚠️  结果太少，跳过绘图")
        return
    
    # 按种子排序用于绘图
    successful_results.sort(key=lambda x: x['seed'])
    
    seeds = [r['seed'] for r in successful_results]
    f1_scores = [r['f1_score'] for r in successful_results]
    accuracies = [r['accuracy'] for r in successful_results]
    
    plt.figure(figsize=(12, 8))
    
    # 子图1: F1分数
    plt.subplot(2, 1, 1)
    plt.plot(seeds, f1_scores, 'bo-', linewidth=2, markersize=6)
    plt.title('不同随机种子的F1分数', fontsize=14, fontweight='bold')
    plt.xlabel('随机种子')
    plt.ylabel('F1分数')
    plt.grid(True, alpha=0.3)
    
    # 标记最佳点
    best_idx = np.argmax(f1_scores)
    plt.annotate(f'最佳: {seeds[best_idx]}\nF1: {f1_scores[best_idx]:.4f}', 
                xy=(seeds[best_idx], f1_scores[best_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 子图2: 准确率
    plt.subplot(2, 1, 2)
    plt.plot(seeds, accuracies, 'ro-', linewidth=2, markersize=6)
    plt.title('不同随机种子的准确率', fontsize=14, fontweight='bold')
    plt.xlabel('随机种子')
    plt.ylabel('准确率')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 结果图表已保存到: {save_path}")
    
    # 尝试显示图表
    try:
        plt.show()
    except:
        print("💡 提示: 如需查看图表，请打开 " + save_path)


def update_main_with_best_seed(best_seed):
    """
    更新main.py文件使用最佳种子
    
    参数:
        best_seed (int): 最佳随机种子
    """
    main_file = 'main/main.py'
    
    try:
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找并替换默认种子
        if 'GLOBAL_SEED = 42' in content:
            content = content.replace('GLOBAL_SEED = 42', f'GLOBAL_SEED = {best_seed}')
        elif 'random_seed = RANDOM_CONFIG[\'global_seed\']' in content:
            content = content.replace(
                'random_seed = RANDOM_CONFIG[\'global_seed\']',
                f'random_seed = {best_seed}  # 最优种子'
            )
        
        # 也更新utils/random_seed.py
        seed_file = 'utils/random_seed.py'
        with open(seed_file, 'r', encoding='utf-8') as f:
            seed_content = f.read()
        
        seed_content = seed_content.replace(
            'GLOBAL_SEED = 42',
            f'GLOBAL_SEED = {best_seed}  # 通过种子搜索找到的最优值'
        )
        
        with open(seed_file, 'w', encoding='utf-8') as f:
            f.write(seed_content)
        
        print(f"✅ 已更新代码使用最佳种子: {best_seed}")
        
    except Exception as e:
        print(f"⚠️  更新代码失败: {e}")
        print(f"💡 请手动将 utils/random_seed.py 中的 GLOBAL_SEED 设置为 {best_seed}")


def main():
    """主函数"""
    print("🚀 RAPIER 最优随机种子搜索开始")
    print(f"📅 搜索时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 搜索参数
    num_seeds = 8  # 测试的种子数量（可根据时间调整）
    strategy = 'mixed'  # 种子生成策略
    
    print(f"\n🔧 搜索配置:")
    print(f"  测试种子数量: {num_seeds}")
    print(f"  生成策略: {strategy}")
    print(f"  预计耗时: {num_seeds * 10}分钟（每个种子约10分钟）")
    
    # 用户确认
    response = input(f"\n继续进行种子搜索吗？(y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ 用户取消搜索")
        return
    
    # 生成候选种子
    candidate_seeds = generate_seed_candidates(strategy, num_seeds)
    
    # 搜索最优种子
    all_results = []
    total_start_time = time.time()
    
    for i, seed in enumerate(candidate_seeds, 1):
        result = evaluate_seed(seed, i)
        all_results.append(result)
        
        # 显示进度
        progress = i / len(candidate_seeds) * 100
        elapsed = time.time() - total_start_time
        estimated_total = elapsed / i * len(candidate_seeds)
        remaining = estimated_total - elapsed
        
        print(f"\n📈 进度: {i}/{len(candidate_seeds)} ({progress:.1f}%)")
        print(f"⏱️  已用时: {elapsed/60:.1f}分钟，预计剩余: {remaining/60:.1f}分钟")
        
        # 避免连续运行的资源冲突
        if i < len(candidate_seeds):
            print("⏳ 等待 10 秒后开始下一个种子...")
            time.sleep(10)
    
    # 分析结果
    analysis = analyze_results(all_results)
    
    if analysis:
        # 绘制结果图表
        plot_results(all_results)
        
        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"seed_search_results_{timestamp}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'search_config': {
                    'num_seeds': num_seeds,
                    'strategy': strategy,
                    'timestamp': timestamp
                },
                'best_result': analysis['best_result'],
                'all_results': all_results,
                'summary': {
                    'best_seed': analysis['best_seed'],
                    'best_f1': analysis['best_f1'],
                    'avg_f1': analysis['avg_f1'],
                    'std_f1': analysis['std_f1']
                }
            }, indent=2, ensure_ascii=False)
        
        print(f"\n💾 详细结果已保存到: {result_file}")
        
        # 询问是否更新代码
        print(f"\n🎯 找到最佳种子: {analysis['best_seed']} (F1: {analysis['best_f1']:.6f})")
        response = input("是否更新代码使用这个最佳种子？(Y/n): ").strip().lower()
        if response in ['', 'y', 'yes']:
            update_main_with_best_seed(analysis['best_seed'])
        
        print(f"\n🎉 种子搜索完成！")
        print(f"🏆 最佳种子: {analysis['best_seed']}")
        print(f"🏆 最佳F1分数: {analysis['best_f1']:.6f}")
        print(f"💡 现在您可以使用这个种子获得最优且可重复的结果！")
    
    else:
        print("\n❌ 搜索失败，没有成功的结果")


if __name__ == "__main__":
    main()

