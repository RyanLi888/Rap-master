#!/usr/bin/env python3
"""
RAPIER 系统完整性检查脚本
========================

检查RAPIER系统的所有组件是否正确配置和可用。

功能:
1. 检查所有模块导入
2. 验证最优配置
3. 检查文件结构
4. 验证种子搜索功能
5. 生成系统状态报告

作者: RAPIER 开发团队
版本: 2.0
"""

import os
import sys
import json
import datetime

def check_module_imports():
    """检查模块导入"""
    print("🔍 检查模块导入...")
    results = {}
    
    # 检查路径设置
    sys.path.append('.')
    sys.path.append('utils')
    
    # 检查随机种子模块
    try:
        from random_seed import set_random_seed, RANDOM_CONFIG, deterministic_shuffle, create_deterministic_dataloader, get_deterministic_random_int
        results['random_seed'] = {'status': 'success', 'config': RANDOM_CONFIG}
        print("  ✅ random_seed模块 - 正常")
    except Exception as e:
        results['random_seed'] = {'status': 'failed', 'error': str(e)}
        print(f"  ❌ random_seed模块 - 失败: {e}")
    
    # 检查核心模块
    modules = ['AE', 'MADE', 'Classifier']
    for module_name in modules:
        try:
            module = __import__(module_name)
            results[module_name] = {'status': 'success'}
            print(f"  ✅ {module_name}模块 - 正常")
        except Exception as e:
            results[module_name] = {'status': 'failed', 'error': str(e)}
            print(f"  ❌ {module_name}模块 - 失败: {e}")
    
    return results

def check_file_structure():
    """检查文件结构"""
    print("\n📁 检查文件结构...")
    
    required_dirs = [
        'main', 'AE', 'MADE', 'Classifier', 'utils', 
        'data', 'seed_search'
    ]
    
    required_files = [
        'main/main.py',
        'utils/random_seed.py', 
        'seed_search/global_seed_search.py',
        'seed_search/module_independent_search.py',
        'seed_search/advanced_seed_search.py',
        'README.md',
        'PROJECT_OVERVIEW.md'
    ]
    
    results = {'dirs': {}, 'files': {}}
    
    # 检查目录
    for dir_name in required_dirs:
        exists = os.path.exists(dir_name)
        results['dirs'][dir_name] = exists
        status = "✅" if exists else "❌"
        print(f"  {status} {dir_name}/")
    
    # 检查文件
    for file_path in required_files:
        exists = os.path.exists(file_path)
        results['files'][file_path] = exists
        status = "✅" if exists else "❌"
        print(f"  {status} {file_path}")
    
    return results

def check_optimal_configuration():
    """检查最优配置"""
    print("\n⚙️ 检查最优配置...")
    
    try:
        from random_seed import RANDOM_CONFIG
        
        # 检查种子配置
        expected_config = {
            'global_seed': 2024,
            'ae_seed': 290984,
            'made_seed': 290713,
            'classifier_seed': 19616,
            'generation_seed': 61592
        }
        
        config_correct = True
        for key, expected_value in expected_config.items():
            actual_value = RANDOM_CONFIG.get(key)
            if actual_value == expected_value:
                print(f"  ✅ {key}: {actual_value}")
            else:
                print(f"  ❌ {key}: 期望{expected_value}, 实际{actual_value}")
                config_correct = False
        
        # 检查分类器默认参数
        with open('Classifier/classify.py', 'r') as f:
            content = f.read()
            if 'parallel=1' in content:
                print("  ✅ 分类器默认parallel=1")
                parallel_correct = True
            else:
                print("  ❌ 分类器默认parallel配置错误")
                parallel_correct = False
        
        return {
            'seed_config': config_correct,
            'parallel_config': parallel_correct,
            'overall': config_correct and parallel_correct
        }
        
    except Exception as e:
        print(f"  ❌ 配置检查失败: {e}")
        return {'overall': False, 'error': str(e)}

def check_seed_search_functionality():
    """检查种子搜索功能"""
    print("\n🔍 检查种子搜索功能...")
    
    try:
        # 检查种子搜索文件
        seed_search_file = 'seed_search/optimal_seed_finder.py'
        if os.path.exists(seed_search_file):
            print(f"  ✅ 种子搜索文件存在")
            
            # 检查输出目录
            seed_data_dir = 'seed_data'
            if os.path.exists(seed_data_dir):
                print(f"  ✅ 种子数据目录存在")
                
                # 检查是否有历史结果
                json_files = [f for f in os.listdir(seed_data_dir) if f.endswith('.json')]
                print(f"  📊 历史结果文件: {len(json_files)}个")
                
                return {'status': 'success', 'history_files': len(json_files)}
            else:
                print(f"  ⚠️  种子数据目录不存在，将在首次使用时创建")
                return {'status': 'success', 'history_files': 0}
        else:
            print(f"  ❌ 种子搜索文件不存在")
            return {'status': 'failed', 'error': 'Missing seed search file'}
            
    except Exception as e:
        print(f"  ❌ 种子搜索检查失败: {e}")
        return {'status': 'failed', 'error': str(e)}

def generate_system_report():
    """生成系统状态报告"""
    print("\n📋 生成系统状态报告...")
    
    # 执行所有检查
    module_results = check_module_imports()
    structure_results = check_file_structure()
    config_results = check_optimal_configuration()
    seed_search_results = check_seed_search_functionality()
    
    # 汇总结果
    report = {
        'timestamp': datetime.datetime.now().isoformat(),
        'system_status': 'healthy',
        'checks': {
            'module_imports': module_results,
            'file_structure': structure_results,
            'optimal_configuration': config_results,
            'seed_search': seed_search_results
        }
    }
    
    # 判断整体状态
    failed_modules = [k for k, v in module_results.items() if v.get('status') == 'failed']
    missing_dirs = [k for k, v in structure_results['dirs'].items() if not v]
    missing_files = [k for k, v in structure_results['files'].items() if not v]
    
    if failed_modules or missing_dirs or missing_files or not config_results.get('overall', False):
        report['system_status'] = 'issues_found'
    
    # 保存报告
    report_file = 'system_integrity_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 系统状态: {report['system_status']}")
    if failed_modules:
        print(f"❌ 模块导入失败: {failed_modules}")
    if missing_dirs:
        print(f"❌ 缺失目录: {missing_dirs}")
    if missing_files:
        print(f"❌ 缺失文件: {missing_files}")
    
    if report['system_status'] == 'healthy':
        print("🎉 系统状态良好，所有组件正常!")
    
    print(f"💾 详细报告已保存到: {report_file}")
    
    return report

def main():
    """主函数"""
    print("🔍 RAPIER 系统完整性检查")
    print("=" * 50)
    
    report = generate_system_report()
    
    print("\n" + "=" * 50)
    print("检查完成!")
    
    return report

if __name__ == '__main__':
    main()
