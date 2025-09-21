"""
RAPIER 模块独立种子搜索器
========================

独立搜索每个模块的最优种子，然后组合成全局最优配置。
这种方法可以大大减少搜索空间，提高效率。

搜索策略:
1. 固定其他模块种子，只搜索目标模块种子
2. 找到每个模块的最优种子
3. 组合所有模块最优种子
4. 验证组合效果

作者: RAPIER 开发团队
版本: 1.0 (模块独立版)
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

class ModuleIndependentSearcher:
    """模块独立种子搜索器"""
    
    def __init__(self, output_dir='../seed_data'):
        self.output_dir = output_dir
        self.results_file = os.path.join(output_dir, 'module_independent_results.json')
        self.best_config_file = os.path.join(output_dir, 'best_module_config.json')
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载历史结果
        self.history = self.load_history()
        self.best_result = self.load_best_result()
        
        # 模块定义
        self.modules = {
            'global': {'name': '全局种子', 'key': 'global_seed'},
            'ae': {'name': 'AE模块', 'key': 'ae_seed'},
            'made': {'name': 'MADE模块', 'key': 'made_seed'},
            'classifier': {'name': '分类器模块', 'key': 'classifier_seed'},
            'generation': {'name': '生成器模块', 'key': 'generation_seed'}
        }
    
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
    
    def create_base_config(self):
        """创建基础配置"""
        return {
            'global_seed': 2024,
            'ae_seed': 290984,
            'made_seed': 290713,
            'classifier_seed': 19616,
            'generation_seed': 61592
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
    
    def test_seed_config(self, seed_config, test_name=""):
        """测试特定种子配置的性能"""
        config_str = f"G{seed_config['global_seed']}_A{seed_config['ae_seed']}_M{seed_config['made_seed']}_C{seed_config['classifier_seed']}_Gen{seed_config['generation_seed']}"
        print(f"🔍 测试配置: {test_name} - {config_str}")
        
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
                'test_name': test_name,
                'config_string': config_str
            }
            
            print(f"✅ {test_name}: F1={f1_score:.4f}, 用时={elapsed_time:.1f}s")
            return result
            
        except Exception as e:
            print(f"❌ {test_name} 失败: {str(e)}")
            return {
                'seed_config': seed_config,
                'f1_score': 0.0,
                'elapsed_time': time.time() - start_time,
                'timestamp': datetime.datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e),
                'test_name': test_name,
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
    
    def search_module_independent(self, module_candidates):
        """
        独立搜索每个模块的最优种子
        
        参数:
            module_candidates (dict): 各模块的候选种子
                例如: {
                    'global': [2024, 2025, 2026],
                    'ae': [290000, 290500, 291000],
                    'made': [290000, 290500, 291000],
                    'classifier': [19000, 19500, 20000],
                    'generation': [61000, 61500, 62000]
                }
        """
        print("🧠 模块独立种子搜索")
        print("=" * 60)
        
        # 基础配置
        base_config = self.create_base_config()
        module_results = {}
        
        # 为每个模块独立搜索
        for module_name, module_info in self.modules.items():
            print(f"\n🔍 搜索 {module_info['name']} 的最优种子...")
            print("-" * 40)
            
            candidates = module_candidates.get(module_name, [base_config[module_info['key']]])
            best_f1 = 0.0
            best_seed = None
            
            for seed in candidates:
                # 创建测试配置：只改变目标模块的种子
                test_config = base_config.copy()
                test_config[module_info['key']] = seed
                
                test_name = f"{module_info['name']}_seed_{seed}"
                result = self.test_seed_config(test_config, test_name)
                
                # 记录结果
                self.history.append(result)
                
                # 更新最佳结果
                if result['status'] == 'success' and result['f1_score'] > best_f1:
                    best_f1 = result['f1_score']
                    best_seed = seed
                    print(f"🎉 发现更好的种子: {seed} (F1={best_f1:.4f})")
            
            # 保存该模块的最佳结果
            module_results[module_name] = {
                'best_seed': best_seed,
                'best_f1': best_f1,
                'candidates_tested': len(candidates)
            }
            
            print(f"✅ {module_info['name']} 搜索完成:")
            print(f"   最佳种子: {best_seed}")
            print(f"   最佳F1: {best_f1:.4f}")
            print(f"   测试数量: {len(candidates)}")
        
        # 组合所有模块的最佳种子
        print(f"\n🔗 组合各模块最优种子...")
        print("-" * 40)
        
        optimal_config = base_config.copy()
        for module_name, module_info in self.modules.items():
            optimal_config[module_info['key']] = module_results[module_name]['best_seed']
        
        # 验证组合效果
        print("🧪 验证组合配置效果...")
        combination_result = self.test_seed_config(optimal_config, "组合最优配置")
        self.history.append(combination_result)
        
        # 保存结果
        final_result = {
            'module_results': module_results,
            'optimal_config': optimal_config,
            'combination_result': combination_result,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # 检查是否是新的最佳结果
        if combination_result['status'] == 'success':
            if not self.best_result or combination_result['f1_score'] > self.best_result['f1_score']:
                self.best_result = combination_result.copy()
                print(f"🎉 发现新的全局最佳配置: F1={combination_result['f1_score']:.4f}")
                self.save_best_result()
        
        self.save_history()
        
        return final_result
    
    def print_module_results(self, results):
        """打印模块搜索结果"""
        print("\n📊 模块独立搜索结果")
        print("=" * 60)
        
        module_results = results['module_results']
        
        print(f"{'模块':<12} {'最佳种子':<10} {'最佳F1':<8} {'测试数量':<8}")
        print("-" * 50)
        
        for module_name, module_info in self.modules.items():
            result = module_results[module_name]
            print(f"{module_info['name']:<12} {result['best_seed']:<10} {result['best_f1']:<8.4f} {result['candidates_tested']:<8}")
        
        print("\n🔗 组合配置:")
        optimal_config = results['optimal_config']
        print(f"   全局种子: {optimal_config['global_seed']}")
        print(f"   AE种子: {optimal_config['ae_seed']}")
        print(f"   MADE种子: {optimal_config['made_seed']}")
        print(f"   分类器种子: {optimal_config['classifier_seed']}")
        print(f"   生成器种子: {optimal_config['generation_seed']}")
        
        combination_result = results['combination_result']
        if combination_result['status'] == 'success':
            print(f"   组合F1: {combination_result['f1_score']:.4f}")
        else:
            print(f"   组合状态: 失败 ({combination_result.get('error', '未知错误')})")
    
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

# 预定义的搜索策略
MODULE_SEARCH_STRATEGIES = {
    'quick': {
        'global': [2024, 2025, 2026],
        'ae': [290000, 290500, 291000],
        'made': [290000, 290500, 291000],
        'classifier': [19000, 19500, 20000],
        'generation': [61000, 61500, 62000],
        'description': '快速搜索 - 每个模块测试3个种子'
    },
    'comprehensive': {
        'global': [2024, 2025, 2026, 2027, 2028],
        'ae': [290000, 290250, 290500, 290750, 291000],
        'made': [290000, 290250, 290500, 290750, 291000],
        'classifier': [19000, 19250, 19500, 19750, 20000],
        'generation': [61000, 61250, 61500, 61750, 62000],
        'description': '全面搜索 - 每个模块测试5个种子'
    },
    'extensive': {
        'global': [2024, 2025, 2026, 2027, 2028, 2029, 2030],
        'ae': [290000, 290200, 290400, 290600, 290800, 291000, 291200],
        'made': [290000, 290200, 290400, 290600, 290800, 291000, 291200],
        'classifier': [19000, 19200, 19400, 19600, 19800, 20000, 20200],
        'generation': [61000, 61200, 61400, 61600, 61800, 62000, 62200],
        'description': '广泛搜索 - 每个模块测试7个种子'
    }
}

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAPIER模块独立种子搜索器')
    parser.add_argument('--strategy', choices=['quick', 'comprehensive', 'extensive'], default='quick',
                       help='搜索策略')
    parser.add_argument('--output', type=str, default='../seed_data',
                       help='输出目录')
    parser.add_argument('--best', action='store_true',
                       help='显示最佳配置')
    parser.add_argument('--custom', nargs='+', type=int,
                       help='自定义全局种子列表')
    
    args = parser.parse_args()
    
    searcher = ModuleIndependentSearcher(args.output)
    
    if args.best:
        searcher.print_best_result()
        return
    
    print("🧠 RAPIER 模块独立种子搜索器")
    print("=" * 50)
    
    # 选择搜索策略
    if args.custom:
        strategy = {
            'global': args.custom,
            'ae': [290000, 290500, 291000],
            'made': [290000, 290500, 291000],
            'classifier': [19000, 19500, 20000],
            'generation': [61000, 61500, 62000]
        }
        print(f"📋 使用自定义策略: 全局种子 {args.custom}")
    else:
        strategy = MODULE_SEARCH_STRATEGIES[args.strategy]
        print(f"📋 使用策略: {strategy['description']}")
    
    # 执行搜索
    results = searcher.search_module_independent(strategy)
    
    # 打印结果
    searcher.print_module_results(results)
    
    print(f"\n💾 搜索结果已保存到: {searcher.results_file}")
    if searcher.best_result:
        print(f"💾 最佳配置已保存到: {searcher.best_config_file}")

if __name__ == '__main__':
    main()
