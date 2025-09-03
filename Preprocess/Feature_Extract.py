"""
网络流量特征提取模块
==================

本文件实现了从PCAP文件中提取网络流量特征的功能，包括：
1. 流量识别和分类
2. 突发流量检测
3. 序列数据生成
4. 特征文件输出

该模块主要用于预处理网络流量数据，为后续的机器学习模型提供输入。

作者: RAPIER 开发团队
版本: 1.0
"""

import os
import sys
import dpkt
import socket

from xgboost import train
workspace=sys.path[0]

class one_flow(object):
    """
    单个网络流量对象
    
    该类表示一个完整的网络连接流，包含源IP、目标IP、端口等信息，
    以及该流中的所有数据包突发信息。
    """
    
    def __init__(self,pkt_id,timestamp,direction,pkt_length):
        """
        初始化流量对象
        
        参数:
            pkt_id (str): 数据包标识符
            timestamp (float): 时间戳
            direction (int): 数据包方向（1表示出站，-1表示入站）
            pkt_length (int): 数据包长度
        """
        # 固定属性
        self.pkt_id = pkt_id
        
        # 解析数据包标识符获取详细信息
        detailed_info = pkt_id.split("_")
        self.client_ip = detailed_info[0]      # 客户端IP
        self.client_port = int(detailed_info[1])  # 客户端端口
        self.outside_ip = detailed_info[2]     # 外部IP
        self.outside_port = int(detailed_info[3])  # 外部端口
        
        self.start_time = timestamp  # 流开始时间
        
        # 可更新属性
        self.last_time = timestamp   # 最后数据包时间
        self.pkt_count = 1           # 数据包计数
        
        # 突发流量列表
        self.burst_list = [one_burst(timestamp, direction, pkt_length)]
    
    def update(self,timestamp,direction,pkt_length):
        """
        更新流量信息
        
        参数:
            timestamp (float): 新数据包时间戳
            direction (int): 新数据包方向
            pkt_length (int): 新数据包长度
        """
        self.pkt_count += 1
        self.last_time = timestamp
        
        # 如果方向改变，创建新的突发流量
        if self.burst_list[-1].direction != direction:
            self.burst_list.append(one_burst(timestamp,direction,pkt_length))
        else:
            # 否则更新现有突发流量
            self.burst_list[-1].update(timestamp,pkt_length)
            
class one_burst(object):
    """
    单个突发流量对象
    
    该类表示一个连续的数据包突发，包含方向、时间、计数和总长度信息。
    """
    
    def __init__(self,timestamp,direction,pkt_length):
        """
        初始化突发流量对象
        
        参数:
            timestamp (float): 突发开始时间戳
            direction (int): 数据包方向
            pkt_length (int): 数据包长度
        """
        # 固定属性
        self.direction = direction
        self.start_time = timestamp
        
        # 可更新属性
        self.last_time = timestamp
        self.pkt_count = 1
        self.pkt_length = pkt_length
        
    def update(self,timestamp,pkt_length):
        """
        更新突发流量信息
        
        参数:
            timestamp (float): 新时间戳
            pkt_length (int): 新数据包长度
        """
        self.last_time = timestamp
        self.pkt_count += 1
        self.pkt_length += pkt_length
		
def inet_to_str(inet):
    """
    将网络字节序的IP地址转换为字符串
    
    参数:
        inet: 网络字节序的IP地址
        
    返回:
        str: 点分十进制IP地址字符串
    """
    return socket.inet_ntop(socket.AF_INET, inet)

def get_burst_based_flows(pcap):
    """
    从PCAP数据中提取基于突发的流量
    
    该函数解析PCAP文件，识别TCP连接，并将数据包组织成流量和突发结构。
    
    参数:
        pcap: PCAP读取器对象
        
    返回:
        list: 流量对象列表
    """
    current_flows = dict()
    
    # 遍历所有数据包
    for i, (timestamp, buf) in enumerate(pcap):
        try:
            # 解析以太网帧
            eth = dpkt.ethernet.Ethernet(buf)
        except Exception as e:
            print(e)
            continue
        
        # 处理不同的链路层协议
        if not isinstance(eth.data, dpkt.ip.IP):
            eth = dpkt.sll.SLL(buf)
            if not isinstance(eth.data, dpkt.ip.IP):
                continue
		
        # 获取IP层信息
        ip = eth.data
        pkt_length = ip.len
		
        # 获取源和目标IP地址
        src_ip = inet_to_str(ip.src)
        dst_ip = inet_to_str(ip.dst)
    
        # 只处理TCP协议
        if not isinstance(ip.data, dpkt.tcp.TCP):
            continue

        # 获取TCP端口信息
        tcp = ip.data
        srcport = tcp.sport
        dstport = tcp.dport
        direction = None
        
        # 识别HTTPS流量（端口443）
        if dstport == 443:
            direction = -1  # 入站流量
            pkt_id = src_ip+"_"+str(srcport)+"_"+dst_ip+"_"+str(dstport)
        elif srcport == 443:
            direction = 1   # 出站流量
            pkt_id = dst_ip+"_"+str(dstport)+"_"+src_ip+"_"+str(srcport)
        else:
            continue
        
        # 更新或创建流量对象
        if pkt_id in current_flows:
            current_flows[pkt_id].update(timestamp,direction,pkt_length)
        else:
            current_flows[pkt_id] = one_flow(pkt_id,timestamp,direction,pkt_length)

    return list(current_flows.values())

def get_flows(file):
    """
    从文件中获取流量信息
    
    参数:
        file (str): PCAP文件路径
        
    返回:
        list: 流量对象列表
    """
    with open(file,"rb") as input:
        pcap = dpkt.pcap.Reader(input)
        all_flows = get_burst_based_flows(pcap)
        return all_flows

def generate_sequence_data(all_files_flows, output_file, label_file):
    """
    生成序列数据和标签
    
    该函数将流量对象转换为序列特征数据，用于机器学习模型训练。
    
    参数:
        all_files_flows (list): 所有文件的流量列表
        output_file (str): 特征输出文件路径
        label_file (str): 标签输出文件路径
    """
    output_features = []
    output_labels = []
    
    for flow in all_files_flows:
        one_flow = []
        client_ip = flow.client_ip
        outside_ip = flow.outside_ip
        label = client_ip + '-' + outside_ip
        
        # 生成累积序列数据
        for index,burst in enumerate(flow.burst_list):
            if index != 0:
                # 计算累积值
                current_cumulative = one_flow[-1] + (burst.pkt_length * burst.direction)
                one_flow.append(current_cumulative)
            else:
                # 第一个突发流量
                one_flow.append(burst.pkt_length * burst.direction)
        
        # 转换为字符串格式
        one_flow = [str(value) for value in one_flow]
        one_line = ",".join(one_flow)
        output_features.append(one_line)
        output_labels.append(label)
    
    # 写入文件
    write_into_files(output_features, output_file)
    write_into_files(output_labels, label_file)

def write_into_files(output_features,output_file):
    """
    将特征数据写入文件
    
    参数:
        output_features (list): 特征数据列表
        output_file (str): 输出文件路径
    """
    with open(output_file,"w") as write_fp:
        output_features = [value+"\n" for value in output_features]
        write_fp.writelines(output_features)

def main(input_dir, output_path, suffix):
    """
    主函数：处理指定目录下的所有PCAP文件
    
    参数:
        input_dir (str): 输入目录路径
        output_path (str): 输出文件路径
        suffix (str): 文件扩展名（如pcap, pcapng）
    """
    print(f"开始处理目录: {input_dir}")
    
    # 查找所有指定扩展名的文件
    pcap_filedir = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file[-len(suffix)-1:] == '.'+suffix:
                pcap_filedir.append(os.path.join(root, file))

    files = pcap_filedir
    print(f"找到 {len(files)} 个PCAP文件")
    
    # 处理所有文件
    all_files_flows = []
    for file in files:
        try:
            flows_of_file = get_flows(file)
        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")
            pass
            
        if flows_of_file == False:  # 错误记录
            print(file, "Critical Error2")
            continue
            
        if len(flows_of_file) <= 0:
            continue
            
        all_files_flows += flows_of_file
 
    print(f"总共提取了 {len(all_files_flows)} 个流量")
    
    # 生成序列数据
    generate_sequence_data(all_files_flows, output_path, output_path + '_labels')
    print("特征提取完成！")

if __name__ == "__main__":
    # 命令行参数：脚本名、输入目录、输出路径、文件扩展名
    _, input_dir, output_path, suffix = sys.argv
    main(input_dir, output_path, suffix)
    
