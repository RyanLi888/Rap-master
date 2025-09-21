"""
RAPIER 全局种子搜索器
==================

专门用于全局种子搜索的独立工具。
这是最简单、最快速的种子搜索方法，适合快速验证和初步优化。

搜索策略:
1. 使用全局种子派生其他模块种子
2. 测试不同全局种子的效果
3. 找到最优全局种子

作者: RAPIER 开发团队
版本: 1.0 (全局专用版)
"""

import os
import sys
import json
import time
import datetime
import shutil
import numpy as np
from typing import Dict, List, Tuple, Optional

# 添加父目录到系统路径
sys.path.append('..')
sys.path.append('../main')
import MADE
import Classifier
import AE

# 导入随机种子控制模块
sys.path.append('../utils')
try:
    from random_seed import set_random_seed, RANDOM_CONFIG
    SEED_CONTROL_AVAILABLE = True
except ImportError:
    SEED_CONTROL_AVAILABLE = False
    print("⚠️  警告：随机种子控制模块不可用")

class GlobalSeedSearcher:
    """全局种子搜索器"""
    
    def __init__(self, output_dir='../seed_data'):
        self.output_dir = output_dir
        self.results_file = os.path.join(output_dir, 'global_seed_results.json')
        self.best_config_file = os.path.join(output_dir, 'best_global_config.json')
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载历史结果
        self.history = self.load_history()
        self.best_result = self.load_best_result()
    
    def load_history(self):
        """加载搜索历史"""
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def load_best_result(self):
        """加载最佳结果"""
        if os.path.exists(self.best_config_file):
            try:
                with open(self.best_config_file, 'r') as f:
                    return json.load(f)
            except:
                return None
        return None
    
    def save_history(self):
        """保存搜索历史"""
        with open(self.results_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def save_best_result(self):
        """保存最佳结果"""
        if self.best_result:
            with open(self.best_config_file, 'w') as f:
                json.dump(self.best_result, f, indent=2)
    
    def create_seed_config(self, global_seed):
        """
        创建种子配置：只改变全局种子，其他种子保持原有配置
        
        这样可以在保持其他模块种子不变的情况下，测试不同全局种子的效果
        """
        return {
            'global_seed': global_seed,
            'ae_seed': 290984,      # 保持原有配置
            'made_seed': 290713,    # 保持原有配置
            'classifier_seed': 19616, # 保持原有配置
            'generation_seed': 61592  # 保持原有配置
        }
    
    def apply_seed_config(self, seed_config):
        """应用种子配置到random_seed模块"""
        random_seed_file = '../utils/random_seed.py'
        
        # 读取当前文件
        with open(random_seed_file, 'r') as f:
            content = f.read()
        
        # 备份原始文件
        backup_file = random_seed_file + '.backup'
        with open(backup_file, 'w') as f:
            f.write(content)
        
        # 替换配置
        new_content = content
        for key, value in seed_config.items():
            if key == 'global_seed':
                new_content = new_content.replace(f'GLOBAL_SEED = 2024', f'GLOBAL_SEED = {value}')
            elif key == 'ae_seed':
                new_content = new_content.replace(f'AE_SEED = 290984', f'AE_SEED = {value}')
            elif key == 'made_seed':
                new_content = new_content.replace(f'MADE_SEED = 290713', f'MADE_SEED = {value}')
            elif key == 'classifier_seed':
                new_content = new_content.replace(f'CLASSIFIER_SEED = 19616', f'CLASSIFIER_SEED = {value}')
            elif key == 'generation_seed':
                new_content = new_content.replace(f'GENERATION_SEED = 61592', f'GENERATION_SEED = {value}')
        
        # 写入新配置
        with open(random_seed_file, 'w') as f:
            f.write(new_content)
        
        return backup_file
    
    def restore_seed_config(self, backup_file):
        """恢复原始种子配置"""
        random_seed_file = '../utils/random_seed.py'
        
        if os.path.exists(backup_file):
            with open(backup_file, 'r') as f:
                content = f.read()
            
            with open(random_seed_file, 'w') as f:
                f.write(content)
            
            os.remove(backup_file)
    
    def test_global_seed(self, global_seed):
        """测试特定全局种子的性能"""
        seed_config = self.create_seed_config(global_seed)
        config_str = f"G{global_seed}_A{seed_config['ae_seed']}_M{seed_config['made_seed']}_C{seed_config['classifier_seed']}_Gen{seed_config['generation_seed']}"
        print(f"🔍 测试全局种子: {global_seed} (其他种子保持原有配置)")
        
        start_time = time.time()
        backup_file = None
        
        try:
            # 应用种子配置
            backup_file = self.apply_seed_config(seed_config)
            
            # 创建临时工作目录
            temp_dir = os.path.join(self.output_dir, f'temp_global_{global_seed}_{int(time.time())}')
            data_dir = '../data/data'
            feat_dir = os.path.join(temp_dir, 'feat')
            model_dir = os.path.join(temp_dir, 'model')
            made_dir = os.path.join(temp_dir, 'made')
            result_dir = os.path.join(temp_dir, 'result')
            cuda = 0
            
            # 清理并创建目录
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            for dir_path in [feat_dir, model_dir, made_dir, result_dir]:
                os.makedirs(dir_path, exist_ok=True)
            
            # 重新导入模块以应用新配置
            import importlib
            sys.path.insert(0, '../utils')
            random_seed_module = importlib.import_module('random_seed')
            importlib.reload(random_seed_module)
            
            # 设置随机种子
            random_seed_module.set_random_seed(global_seed)
            
            # 执行完整的RAPIER流程
            # 阶段1: AE训练和特征提取
            AE.train.main(data_dir, model_dir, cuda)
            AE.get_feat.main(data_dir, model_dir, feat_dir, 'be', cuda)
            AE.get_feat.main(data_dir, model_dir, feat_dir, 'ma', cuda)
            AE.get_feat.main(data_dir, model_dir, feat_dir, 'test', cuda)
            
            # 阶段2: MADE数据清理
            TRAIN = 'be'
            MADE.train_epochs.main(feat_dir, model_dir, made_dir, TRAIN, cuda, '20')
            MADE.get_clean_epochs.main(feat_dir, made_dir, '0.5', TRAIN)
            MADE.final_predict.main(feat_dir, result_dir)
            
            # 阶段3: GAN生成
            for index in range(5):
                self._generate_single(feat_dir, model_dir, made_dir, index, cuda)
            
            # 阶段4: 分类器预测 (使用最优parallel=1)
            TRAIN = 'corrected'
            f1_score = Classifier.classify.predict_only(feat_dir, model_dir, result_dir, TRAIN, cuda, parallel=1)
            
            elapsed_time = time.time() - start_time
            
            # 清理临时目录
            shutil.rmtree(temp_dir)
            
            result = {
                'global_seed': global_seed,
                'seed_config': seed_config,
                'f1_score': float(f1_score),
                'elapsed_time': elapsed_time,
                'timestamp': datetime.datetime.now().isoformat(),
                'status': 'success',
                'config_string': config_str
            }
            
            print(f"✅ 全局种子 {global_seed}: F1={f1_score:.4f}, 用时={elapsed_time:.1f}s")
            return result
            
        except Exception as e:
            print(f"❌ 全局种子 {global_seed} 失败: {str(e)}")
            return {
                'global_seed': global_seed,
                'seed_config': seed_config,
                'f1_score': 0.0,
                'elapsed_time': time.time() - start_time,
                'timestamp': datetime.datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e),
                'config_string': config_str
            }
        finally:
            # 恢复原始配置
            if backup_file:
                self.restore_seed_config(backup_file)
    
    def _generate_single(self, feat_dir, model_dir, made_dir, index, cuda):
        """生成单个对抗样本"""
        TRAIN_be = 'be_corrected'
        TRAIN_ma = 'ma_corrected'
        TRAIN = 'corrected'
        
        MADE.train.main(feat_dir, model_dir, TRAIN_be, cuda, '-30')
        MADE.train.main(feat_dir, model_dir, TRAIN_ma, cuda, '-30')
        
        MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_be, TRAIN_be, cuda)
        MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_be, TRAIN_ma, cuda)
        MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_ma, TRAIN_ma, cuda)
        MADE.predict.main(feat_dir, model_dir, made_dir, TRAIN_ma, TRAIN_be, cuda)
        
        MADE.train_gen_GAN.main(feat_dir, model_dir, made_dir, TRAIN, cuda)
        MADE.generate_GAN.main(feat_dir, model_dir, TRAIN, index, cuda)
    
    def search_global_seeds(self, global_seed_candidates):
        """
        搜索最优全局种子
        
        参数:
            global_seed_candidates (list): 全局种子候选列表
        """
        print("🌍 RAPIER 全局种子搜索器")
        print("=" * 50)
        print(f"📊 候选全局种子: {global_seed_candidates}")
        print(f"🔢 总共需要测试: {len(global_seed_candidates)} 个种子")
        
        results = []
        best_f1 = 0.0
        best_seed = None
        
        for i, global_seed in enumerate(global_seed_candidates, 1):
            print(f"\n进度: {i}/{len(global_seed_candidates)}")
            
            result = self.test_global_seed(global_seed)
            results.append(result)
            self.history.append(result)
            
            # 检查是否是新的最佳结果
            if result['status'] == 'success':
                if result['f1_score'] > best_f1:
                    best_f1 = result['f1_score']
                    best_seed = global_seed
                    self.best_result = result.copy()
                    print(f"🎉 发现新的最佳全局种子: {global_seed} (F1={best_f1:.4f})")
                    self.save_best_result()
            
            # 定期保存历史
            if i % 3 == 0:
                self.save_history()
        
        self.save_history()
        
        # 打印最终结果
        self.print_search_summary(results, best_seed, best_f1)
        
        return results
    
    def print_search_summary(self, results, best_seed, best_f1):
        """打印搜索摘要"""
        successful_results = [r for r in results if r['status'] == 'success']
        
        print(f"\n📊 全局种子搜索结果摘要")
        print("=" * 60)
        print(f"✅ 成功测试: {len(successful_results)}/{len(results)} 个种子")
        print(f"🏆 最佳全局种子: {best_seed}")
        print(f"🎯 最佳F1分数: {best_f1:.4f}")
        
        if successful_results:
            print(f"\n📈 前5名结果:")
            print(f"{'排名':<4} {'全局种子':<8} {'F1分数':<8} {'用时(秒)':<8}")
            print("-" * 35)
            
            # 按F1分数排序
            sorted_results = sorted(successful_results, key=lambda x: x['f1_score'], reverse=True)
            
            for i, result in enumerate(sorted_results[:5], 1):
                print(f"{i:<4} {result['global_seed']:<8} {result['f1_score']:<8.4f} {result['elapsed_time']:<8.1f}")
    
    def print_best_result(self):
        """打印最佳结果"""
        if self.best_result:
            config = self.best_result['seed_config']
            print(f"🏆 最佳全局种子配置:")
            print(f"   全局种子: {self.best_result['global_seed']}")
            print(f"   AE种子: {config['ae_seed']}")
            print(f"   MADE种子: {config['made_seed']}")
            print(f"   分类器种子: {config['classifier_seed']}")
            print(f"   生成器种子: {config['generation_seed']}")
            print(f"   F1分数: {self.best_result['f1_score']:.4f}")
            print(f"   时间: {self.best_result['timestamp'][:19]}")
        else:
            print("❌ 暂无最佳全局种子配置")
    
    def print_history_summary(self, top_n=10):
        """打印搜索历史摘要"""
        successful_results = [r for r in self.history if r['status'] == 'success']
        
        if not successful_results:
            print("📝 暂无成功的搜索记录")
            return
        
        print(f"📝 全局种子搜索历史 (共{len(successful_results)}条成功记录):")
        print("=" * 60)
        
        # 按F1分数排序
        sorted_results = sorted(successful_results, key=lambda x: x['f1_score'], reverse=True)
        
        print(f"{'排名':<4} {'全局种子':<8} {'F1分数':<8} {'用时(秒)':<8} {'时间':<20}")
        print("-" * 60)
        
        for i, result in enumerate(sorted_results[:top_n], 1):
            timestamp = result['timestamp'][:19] if 'timestamp' in result else 'N/A'
            print(f"{i:<4} {result['global_seed']:<8} {result['f1_score']:<8.4f} {result['elapsed_time']:<8.1f} {timestamp:<20}")
    
    def get_seed_config(self, global_seed):
        """获取指定全局种子的完整配置"""
        config = self.create_seed_config(global_seed)
        print(f"🌍 全局种子 {global_seed} 的完整配置:")
        print(f"   全局种子: {config['global_seed']} (可变)")
        print(f"   AE种子: {config['ae_seed']} (固定)")
        print(f"   MADE种子: {config['made_seed']} (固定)")
        print(f"   分类器种子: {config['classifier_seed']} (固定)")
        print(f"   生成器种子: {config['generation_seed']} (固定)")
        print(f"   说明: 只有全局种子会改变，其他种子保持原有配置")

# 预定义的搜索策略
GLOBAL_SEARCH_STRATEGIES = {
    'quick': {
        'seeds': [42, 123, 1234, 2024, 7271],
        'description': '快速搜索 - 测试5个常用全局种子'
    },
    'comprehensive': {
        'seeds': [42, 123, 1234, 2024, 7271, 12345, 54321, 99999, 290713, 291713],
        'description': '全面搜索 - 测试10个全局种子'
    },
    'extensive': {
        'seeds': [42, 123, 1234, 2024, 7271, 12345, 54321, 99999, 290713, 291713, 
                 2025, 2026, 2027, 2028, 2029, 2030, 10000, 20000, 30000, 40000],
        'description': '广泛搜索 - 测试20个全局种子'
    },
    'focused': {
        'seeds': [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030],
        'description': '聚焦搜索 - 测试2020-2030年范围'
    }
}

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAPIER全局种子搜索器')
    parser.add_argument('--strategy', choices=['quick', 'comprehensive', 'extensive', 'focused'], default='quick',
                       help='搜索策略')
    parser.add_argument('--seeds', nargs='+', type=int,
                       help='自定义全局种子列表')
    parser.add_argument('--output', type=str, default='../seed_data',
                       help='输出目录')
    parser.add_argument('--best', action='store_true',
                       help='显示最佳配置')
    parser.add_argument('--history', action='store_true',
                       help='显示搜索历史')
    parser.add_argument('--config', type=int,
                       help='显示指定全局种子的完整配置')
    
    args = parser.parse_args()
    
    searcher = GlobalSeedSearcher(args.output)
    
    if args.best:
        searcher.print_best_result()
        return
    
    if args.history:
        searcher.print_history_summary()
        return
    
    if args.config:
        searcher.get_seed_config(args.config)
        return
    
    print("🌍 RAPIER 全局种子搜索器")
    print("=" * 50)
    
    # 选择搜索策略
    if args.seeds:
        global_seeds = args.seeds
        print(f"📋 使用自定义种子: {global_seeds}")
    else:
        strategy = GLOBAL_SEARCH_STRATEGIES[args.strategy]
        global_seeds = strategy['seeds']
        print(f"📋 使用策略: {strategy['description']}")
    
    # 执行搜索
    results = searcher.search_global_seeds(global_seeds)
    
    print(f"\n💾 搜索结果已保存到: {searcher.results_file}")
    if searcher.best_result:
        print(f"💾 最佳配置已保存到: {searcher.best_config_file}")

if __name__ == '__main__':
    main()
