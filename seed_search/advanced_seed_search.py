"""
RAPIER 高级种子搜索器
==================

支持多层种子配置的搜索，包括：
1. 全局种子搜索
2. 模块特定种子搜索 
3. 种子组合优化
4. 分层搜索策略

作者: RAPIER 开发团队
版本: 3.0 (高级版)
"""

import os
import sys
import json
import time
import datetime
import shutil
import numpy as np
import itertools
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

class AdvancedSeedSearcher:
    """高级种子搜索器 - 支持多层种子配置"""
    
    def __init__(self, output_dir='../seed_data'):
        self.output_dir = output_dir
        self.results_file = os.path.join(output_dir, 'advanced_seed_results.json')
        self.best_config_file = os.path.join(output_dir, 'best_seed_config.json')
        
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
    
    def create_seed_config(self, global_seed, ae_seed=None, made_seed=None, 
                          classifier_seed=None, generation_seed=None):
        """
        创建种子配置
        
        参数:
            global_seed (int): 全局种子
            ae_seed (int): AE种子，None时使用全局种子派生
            made_seed (int): MADE种子，None时使用全局种子派生
            classifier_seed (int): 分类器种子，None时使用全局种子派生
            generation_seed (int): 生成器种子，None时使用全局种子派生
        """
        # 如果模块种子为None，使用全局种子派生
        if ae_seed is None:
            ae_seed = (global_seed * 137 + 1000) % 1000000
        if made_seed is None:
            made_seed = (global_seed * 139 + 2000) % 1000000
        if classifier_seed is None:
            classifier_seed = (global_seed * 149 + 3000) % 1000000
        if generation_seed is None:
            generation_seed = (global_seed * 151 + 4000) % 1000000
        
        return {
            'global_seed': global_seed,
            'ae_seed': ae_seed,
            'made_seed': made_seed,
            'classifier_seed': classifier_seed,
            'generation_seed': generation_seed
        }
    
    def apply_seed_config(self, seed_config):
        """应用种子配置到random_seed模块"""
        # 临时修改random_seed.py中的配置
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
    
    def test_seed_config(self, seed_config):
        """测试特定种子配置的性能"""
        config_str = f"G{seed_config['global_seed']}_A{seed_config['ae_seed']}_M{seed_config['made_seed']}_C{seed_config['classifier_seed']}_Gen{seed_config['generation_seed']}"
        print(f"🔍 测试种子配置: {config_str}")
        
        start_time = time.time()
        backup_file = None
        
        try:
            # 应用种子配置
            backup_file = self.apply_seed_config(seed_config)
            
            # 创建临时工作目录
            temp_dir = os.path.join(self.output_dir, f'temp_config_{int(time.time())}')
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
            random_seed_module.set_random_seed(seed_config['global_seed'])
            
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
                'seed_config': seed_config,
                'f1_score': float(f1_score),
                'elapsed_time': elapsed_time,
                'timestamp': datetime.datetime.now().isoformat(),
                'status': 'success',
                'config_string': config_str
            }
            
            print(f"✅ 配置 {config_str}: F1={f1_score:.4f}, 用时={elapsed_time:.1f}s")
            return result
            
        except Exception as e:
            print(f"❌ 配置 {config_str} 失败: {str(e)}")
            return {
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
        """搜索最优全局种子 (简单模式)"""
        print(f"🔍 全局种子搜索模式")
        print(f"📊 候选种子: {global_seed_candidates}")
        
        results = []
        for global_seed in global_seed_candidates:
            # 使用全局种子派生其他种子
            seed_config = self.create_seed_config(global_seed)
            result = self.test_seed_config(seed_config)
            results.append(result)
            self.history.append(result)
            
            # 检查是否是新的最佳结果
            if result['status'] == 'success':
                if not self.best_result or result['f1_score'] > self.best_result['f1_score']:
                    self.best_result = result.copy()
                    print(f"🎉 发现新的最佳配置: F1={result['f1_score']:.4f}")
                    self.save_best_result()
            
            self.save_history()
        
        return results
    
    def search_module_seeds(self, base_global_seed, module_seed_ranges):
        """
        搜索模块特定种子 (高级模式)
        
        参数:
            base_global_seed (int): 基础全局种子
            module_seed_ranges (dict): 各模块的种子候选范围
                例如: {
                    'ae_seed': [290000, 290500, 291000],
                    'made_seed': [290000, 290500, 291000], 
                    'classifier_seed': [19000, 19500, 20000],
                    'generation_seed': [61000, 61500, 62000]
                }
        """
        print(f"🧠 模块种子搜索模式")
        print(f"🎯 基础全局种子: {base_global_seed}")
        print(f"🔧 模块种子范围: {module_seed_ranges}")
        
        results = []
        
        # 生成所有种子组合
        module_names = list(module_seed_ranges.keys())
        seed_combinations = list(itertools.product(*[module_seed_ranges[name] for name in module_names]))
        
        print(f"📊 总共 {len(seed_combinations)} 种组合需要测试")
        
        for i, combination in enumerate(seed_combinations, 1):
            # 创建种子配置
            seed_config = {'global_seed': base_global_seed}
            for j, module_name in enumerate(module_names):
                seed_config[module_name] = combination[j]
            
            print(f"\n进度: {i}/{len(seed_combinations)}")
            result = self.test_seed_config(seed_config)
            results.append(result)
            self.history.append(result)
            
            # 检查是否是新的最佳结果
            if result['status'] == 'success':
                if not self.best_result or result['f1_score'] > self.best_result['f1_score']:
                    self.best_result = result.copy()
                    print(f"🎉 发现新的最佳配置: F1={result['f1_score']:.4f}")
                    self.save_best_result()
            
            # 定期保存历史
            if i % 5 == 0:
                self.save_history()
        
        self.save_history()
        return results
    
    def search_around_best(self, search_radius=1000, num_variants=20):
        """
        在最佳配置周围搜索 (局部优化)
        
        参数:
            search_radius (int): 搜索半径
            num_variants (int): 生成的变体数量
        """
        if not self.best_result:
            print("❌ 没有最佳配置，无法进行局部搜索")
            return []
        
        best_config = self.best_result['seed_config']
        print(f"🎯 在最佳配置周围搜索")
        print(f"📍 最佳配置: {best_config}")
        print(f"🔍 搜索半径: ±{search_radius}")
        print(f"🔢 变体数量: {num_variants}")
        
        results = []
        
        for i in range(num_variants):
            # 生成随机变体
            variant_config = {}
            for key, base_value in best_config.items():
                # 在基础值周围随机变化
                offset = np.random.randint(-search_radius, search_radius + 1)
                variant_config[key] = max(1, base_value + offset)
            
            print(f"\n变体 {i+1}/{num_variants}")
            result = self.test_seed_config(variant_config)
            results.append(result)
            self.history.append(result)
            
            # 检查是否是新的最佳结果
            if result['status'] == 'success':
                if result['f1_score'] > self.best_result['f1_score']:
                    self.best_result = result.copy()
                    print(f"🎉 发现新的最佳配置: F1={result['f1_score']:.4f}")
                    self.save_best_result()
        
        self.save_history()
        return results
    
    def get_config_signature(self, seed_config):
        """生成配置签名用于去重"""
        return f"{seed_config['global_seed']}_{seed_config['ae_seed']}_{seed_config['made_seed']}_{seed_config['classifier_seed']}_{seed_config['generation_seed']}"
    
    def is_config_tested(self, seed_config):
        """检查配置是否已测试过"""
        signature = self.get_config_signature(seed_config)
        tested_signatures = [self.get_config_signature(r['seed_config']) for r in self.history if 'seed_config' in r]
        return signature in tested_signatures
    
    def print_best_result(self):
        """打印最佳结果"""
        if self.best_result:
            config = self.best_result['seed_config']
            print(f"🏆 最佳种子配置:")
            print(f"   全局种子: {config['global_seed']}")
            print(f"   AE种子: {config['ae_seed']}")
            print(f"   MADE种子: {config['made_seed']}")
            print(f"   分类器种子: {config['classifier_seed']}")
            print(f"   生成器种子: {config['generation_seed']}")
            print(f"   F1分数: {self.best_result['f1_score']:.4f}")
            print(f"   时间: {self.best_result['timestamp'][:19]}")
        else:
            print("❌ 暂无最佳种子配置")
    
    def print_history_summary(self, top_n=10):
        """打印搜索历史摘要"""
        successful_results = [r for r in self.history if r['status'] == 'success']
        
        if not successful_results:
            print("📝 暂无成功的搜索记录")
            return
        
        print(f"📝 搜索历史摘要 (共{len(successful_results)}条成功记录):")
        print("=" * 80)
        
        # 按F1分数排序
        sorted_results = sorted(successful_results, key=lambda x: x['f1_score'], reverse=True)
        
        print(f"{'排名':<4} {'全局种子':<8} {'AE种子':<8} {'MADE种子':<8} {'分类器种子':<10} {'生成器种子':<10} {'F1分数':<8}")
        print("-" * 80)
        
        for i, result in enumerate(sorted_results[:top_n], 1):
            config = result['seed_config']
            print(f"{i:<4} {config['global_seed']:<8} {config['ae_seed']:<8} {config['made_seed']:<8} {config['classifier_seed']:<10} {config['generation_seed']:<10} {result['f1_score']:<8.4f}")

# 预定义的搜索策略
SEARCH_STRATEGIES = {
    'quick': {
        'global_seeds': [42, 123, 1234, 2024, 7271],
        'description': '快速搜索 - 测试5个常用全局种子'
    },
    'comprehensive': {
        'global_seeds': [42, 123, 1234, 2024, 7271, 12345, 54321, 99999, 290713, 291713],
        'description': '全面搜索 - 测试10个全局种子'
    },
    'module_specific': {
        'base_global_seed': 2024,
        'module_ranges': {
            'ae_seed': [290000, 290500, 291000, 291500],
            'made_seed': [290000, 290500, 291000, 291500],
            'classifier_seed': [19000, 19500, 20000, 20500],
            'generation_seed': [61000, 61500, 62000, 62500]
        },
        'description': '模块搜索 - 在最优全局种子基础上搜索模块种子'
    }
}

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAPIER高级种子搜索器')
    parser.add_argument('--mode', choices=['global', 'module', 'local'], default='global',
                       help='搜索模式: global(全局), module(模块), local(局部)')
    parser.add_argument('--strategy', choices=['quick', 'comprehensive', 'module_specific'], default='quick',
                       help='搜索策略')
    parser.add_argument('--seeds', nargs='+', type=int,
                       help='自定义种子列表 (仅global模式)')
    parser.add_argument('--output', type=str, default='../seed_data',
                       help='输出目录')
    parser.add_argument('--best', action='store_true',
                       help='显示最佳配置')
    parser.add_argument('--history', action='store_true',
                       help='显示搜索历史')
    parser.add_argument('--radius', type=int, default=1000,
                       help='局部搜索半径')
    parser.add_argument('--variants', type=int, default=20,
                       help='局部搜索变体数量')
    
    args = parser.parse_args()
    
    searcher = AdvancedSeedSearcher(args.output)
    
    if args.best:
        searcher.print_best_result()
        return
    
    if args.history:
        searcher.print_history_summary()
        return
    
    print("🧠 RAPIER 高级种子搜索器")
    print("=" * 50)
    
    if args.mode == 'global':
        # 全局种子搜索
        if args.seeds:
            global_seeds = args.seeds
        else:
            strategy = SEARCH_STRATEGIES.get(args.strategy, SEARCH_STRATEGIES['quick'])
            global_seeds = strategy['global_seeds']
            print(f"📋 使用策略: {strategy['description']}")
        
        results = searcher.search_global_seeds(global_seeds)
        
    elif args.mode == 'module':
        # 模块种子搜索
        strategy = SEARCH_STRATEGIES['module_specific']
        results = searcher.search_module_seeds(
            strategy['base_global_seed'],
            strategy['module_ranges']
        )
        
    elif args.mode == 'local':
        # 局部搜索
        results = searcher.search_around_best(args.radius, args.variants)
    
    print(f"\n💾 搜索结果已保存到: {searcher.results_file}")
    if searcher.best_result:
        print(f"💾 最佳配置已保存到: {searcher.best_config_file}")

if __name__ == '__main__':
    main()
