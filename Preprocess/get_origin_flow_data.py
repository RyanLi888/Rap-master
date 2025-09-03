"""
序列数据重建模块
================

本文件将由 Feature_Extract.py 生成的逗号分隔的序列文本，
转换为固定长度（50）并进行差分与截断（上限2000）的数值特征，
输出为 `.npy` 格式供后续模型使用。

作者: RAPIER 开发团队
版本: 1.0
"""

import os
import sys
import numpy as np

def get_feat(file):
    """
    读取序列文本并构造长度为50的差分特征
    
    规则：
    - 取前50个累积值，并与前一项做差得到每步长度
    - 绝对值大于等于2000的截断为1999
    - 不足50的序列在尾部补0
    
    参数:
        file (str): 输入文本文件路径
    返回:
        np.ndarray: 形状为 (N, 50) 的整数特征
    """
    try:
        fp = open(file, 'r')
    except:
        return None
    flows = []
    for i, line in enumerate(fp):
        line_s = line.strip().split(';')
        sq = line_s[0].split(',')
        feat = []
        for i in range(50):
            if i >= len(sq):
                feat.append(0)
            else:
                length = abs(int(sq[i]) - (0 if i == 0 else int(sq[i - 1])))
                if length >= 2000:
                    feat.append(1999)
                else:
                    feat.append(length)
        flows.append(feat)
    return np.array(flows, dtype=int)

def main(sequence_data_path, save_dir, data_type):
    """
    主函数：将文本序列转换为npy并保存
    
    参数:
        sequence_data_path (str): 输入序列文本路径
        save_dir (str): 保存目录
        data_type (str): 保存文件名（不含后缀）
    """
    data = get_feat(sequence_data_path)
    np.save(os.path.join(save_dir, data_type), data)

if __name__ == '__main__': 
    # 命令行参数：脚本名、输入序列路径、保存目录、数据类型
    _, sequence_data_path, save_dir, data_type = sys.argv
    main(sequence_data_path, save_dir, data_type)
