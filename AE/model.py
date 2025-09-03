"""
AE (AutoEncoder) 自编码器模型
============================

本文件实现了基于LSTM的自编码器模型，用于脑电图数据的特征提取和重构。
该模型结合了LSTM编码器-解码器架构和GMM（高斯混合模型）估计器。

主要特性：
1. 双向LSTM编码器：将输入序列编码为固定维度的特征向量
2. 双向LSTM解码器：将特征向量重构为原始序列
3. GMM估计器：基于编码特征进行分类预测
4. 支持GPU加速训练

作者: RAPIER 开发团队
版本: 1.0
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class LSTM_AE_GMM(nn.Module):
    """
    基于LSTM的自编码器模型，集成GMM估计器
    
    该模型使用双向LSTM作为编码器和解码器，能够有效处理序列数据，
    并通过重构损失和分类损失进行联合训练。
    
    参数:
        emb_dim (int): 词向量嵌入维度
        input_size (int): 输入序列长度
        hidden_size (int): LSTM隐藏层维度
        dropout (float): Dropout比率
        max_len (int): 最大序列长度
        est_hidden_size (int): 估计器隐藏层维度
        est_output_size (int): 估计器输出维度（类别数）
        device (int): CUDA设备ID
        est_dropout (float): 估计器Dropout比率
        learning_reat (float): 学习率（未使用）
        lambda1 (float): 重构损失权重
        lambda2 (float): 分类损失权重
    """
    
    def __init__(self, emb_dim, input_size, hidden_size, dropout, max_len,
        est_hidden_size, est_output_size, device=0, est_dropout=0.5,
        learning_reat=0.0001, lambda1=0.1, lambda2=0.0001):
        super(LSTM_AE_GMM, self).__init__()

        # 模型参数初始化
        self.max_len = max_len # 最大包长度（已提前处理）
        self.emb_dim = emb_dim # 词向量维度
        self.input_size = input_size # 输入包长度序列长度（AE最终预测长度）
        self.hidden_size = hidden_size # GRU输出维度
        self.dropout = dropout 
        self.device = device

        # 设置CUDA设备
        torch.cuda.set_device(self.device)
        
        # 词向量层 - 将离散的包长度映射为连续向量
        self.embedder = nn.Embedding(self.max_len, self.emb_dim)
        
        # GRU编码器层 - 双向LSTM，提取序列特征
        self.encoders = nn.ModuleList([
            nn.GRU(
                input_size=self.emb_dim,
                hidden_size=self.hidden_size,
                batch_first=True,
                bidirectional=True,
            ).cuda(), 
            nn.GRU(
                input_size=self.hidden_size * 2,  # 双向LSTM输出维度翻倍
                hidden_size=self.hidden_size,
                batch_first=True,
                bidirectional=True,
            ).cuda()
        ])
        
        # GRU解码器层 - 按照FS-Net图所示的双层结构
        self.decoders = nn.ModuleList([
            nn.GRU(
                input_size=self.hidden_size * 4,  # 编码器输出的4倍维度
                hidden_size=self.hidden_size,
                batch_first=True,
                bidirectional=True,
            ).cuda(), 
            nn.GRU(
                input_size=self.hidden_size * 2,
                hidden_size=self.hidden_size,
                batch_first=True,
                bidirectional=True,
            ).cuda()
        ])
        
        # 重构层 - 将解码器输出映射回原始序列长度
        self.rec_fc1 = nn.Linear(4 * self.hidden_size, self.hidden_size)
        self.rec_fc2 = nn.Linear(self.hidden_size, self.max_len)
        self.rec_softmax = nn.Softmax(dim=2)
        
        # 重构损失计算
        self.cross_entropy = nn.CrossEntropyLoss()

        # GMM估计器相关参数和层
        self.est_hidden_size = est_hidden_size
        self.est_output_size = est_output_size
        self.est_dropout = est_dropout
        self.fc1 = nn.Linear(4 * self.hidden_size, self.est_hidden_size)
        self.fc2 = nn.Linear(self.est_hidden_size, self.est_output_size)
        self.est_drop = nn.Dropout(p=self.est_dropout)
        self.softmax = nn.Softmax(dim=1)

        # 训练模式标志
        self.training = False

        # 损失权重参数
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def encode(self, x):
        """
        编码器前向传播
        
        将输入序列通过词向量层和双向LSTM编码器，提取特征表示。
        
        参数:
            x (torch.Tensor): 输入序列张量
            
        返回:
            torch.Tensor: 编码后的特征向量
        """
        torch.cuda.set_device(self.device)
        # 词向量嵌入
        embed_x = self.embedder(x.long())
        if self.training is True:
            embed_x = F.dropout(embed_x)
        
        outputs = [embed_x]
        hs = []
        
        # 通过两层双向LSTM编码器
        for layer in range(2):
            gru = self.encoders[layer]
            output, h = gru(outputs[-1])
            outputs.append(output)
            # 重塑隐藏状态，连接双向输出
            hs.append(torch.transpose(h, 0, 1).reshape(
                -1, 2 * self.hidden_size))
        
        # 连接所有层的输出和隐藏状态
        res = torch.cat(outputs[1:], dim=2)
        res_h = torch.cat(hs, dim=1)
        return res_h
    
    def decode_input(self, x):
        """
        准备解码器输入
        
        将编码器输出扩展为解码器所需的序列长度。
        
        参数:
            x (torch.Tensor): 编码器输出
            
        返回:
            torch.Tensor: 扩展后的解码器输入
        """
        torch.cuda.set_device(self.device)
        y = x.reshape(-1, 1, 4 * self.hidden_size)
        y = y.repeat(1, self.input_size, 1)
        return y
    
    def decode(self, x):
        """
        解码器前向传播
        
        将编码特征通过双向LSTM解码器重构序列。
        
        参数:
            x (torch.Tensor): 解码器输入
            
        返回:
            tuple: (解码器输出, 解码器隐藏状态)
        """
        torch.cuda.set_device(self.device)
        input = x.view(-1, self.input_size, 4 * self.hidden_size)
        outputs = [input]
        hs = []
        
        # 通过两层双向LSTM解码器
        for layer in range(2):
            gru = self.decoders[layer]
            output, h = gru(outputs[-1])
            outputs.append(output)
            hs.append(torch.transpose(h, 0, 1).reshape(
                -1, 2 * self.hidden_size))
        
        # 连接所有层的输出和隐藏状态
        res = torch.cat(outputs[1:], dim=2)
        res_h = torch.cat(hs, dim=1)
        return res, res_h
    
    def reconstruct(self, x, y):
        """
        重构层
        
        将解码器输出映射回原始序列，计算重构损失。
        
        参数:
            x (torch.Tensor): 解码器输出
            y (torch.Tensor): 原始输入序列
            
        返回:
            torch.Tensor: 重构损失
        """
        torch.cuda.set_device(self.device)
        # 通过全连接层重构序列
        x_rec = self.rec_fc2(F.selu(self.rec_fc1(x)))
        # 计算交叉熵损失
        loss = F.cross_entropy(x_rec.view(-1, self.max_len), y.long().view(-1), reduction='none')
        loss = loss.view(-1, self.input_size)
        # 使用掩码处理变长序列
        mask = y.bool()
        loss_ret = torch.sum(loss * mask, dim=1) / torch.sum(mask, dim=1)
        return loss_ret
    
    def estimate(self, x):
        """
        GMM估计器
        
        基于编码特征进行分类预测。
        
        参数:
            x (torch.Tensor): 编码特征
            
        返回:
            torch.Tensor: 分类概率分布
        """
        torch.cuda.set_device(self.device)
        x = x.view(-1, 4 * self.hidden_size)
        res = self.est_drop(F.tanh(self.fc1(x)))
        res = self.softmax(self.fc2(res))
        return res
    
    def feature(self, input):
        """
        特征提取
        
        仅使用编码器提取特征，不进行重构。
        
        参数:
            input (torch.Tensor): 输入序列
            
        返回:
            torch.Tensor: 编码特征
        """
        torch.cuda.set_device(self.device)
        res_encode_h = self.encode(input.float())
        return res_encode_h

    def predict(self, input):
        """
        完整的前向传播
        
        包括编码、解码和重构的完整流程。
        
        参数:
            input (torch.Tensor): 输入序列
            
        返回:
            tuple: (编码特征, 重构损失)
        """
        torch.cuda.set_device(self.device)
        res_encode_h = self.encode(input.float())
        decode_input = self.decode_input(res_encode_h)
        res_decode, res_decode_h = self.decode(decode_input)
        loss_all = self.reconstruct(res_decode, input)
        return res_encode_h, loss_all

    def loss(self, input):
        """
        计算重构损失
        
        参数:
            input (torch.Tensor): 输入序列
            
        返回:
            torch.Tensor: 平均重构损失
        """
        torch.cuda.set_device(self.device)
        _, loss_all = self.predict(input)
        return torch.mean(loss_all, dim=0)
    
    def classify_loss(self, input, labels):
        """
        分类损失（联合训练）
        
        结合重构损失和分类损失进行联合优化。
        
        参数:
            input (torch.Tensor): 输入序列
            labels (torch.Tensor): 真实标签
            
        返回:
            torch.Tensor: 联合损失
        """
        torch.cuda.set_device(self.device)
        feats, rec_loss = self.predict(input)
        score = self.estimate(feats)
        return F.cross_entropy(score, labels) + torch.mean(rec_loss, dim=0)
    
    def classify_loss_1(self, input, labels):
        """
        分类损失（替代版本）
        
        返回未归约的损失，用于更精细的损失控制。
        
        参数:
            input (torch.Tensor): 输入序列
            labels (torch.Tensor): 真实标签
            
        返回:
            torch.Tensor: 未归约的分类损失
        """
        torch.cuda.set_device(self.device)
        feats, rec_loss = self.predict(input)
        score = self.estimate(feats)
        return F.cross_entropy(score, labels, reduce = False) + rec_loss

    def train_mode(self):
        """设置为训练模式"""
        self.training = True
    
    def test_mode(self):
        """设置为测试模式"""
        self.training = False

    def to_cpu(self):
        """将模型转移到CPU"""
        self.device = None
        for encoder in self.encoders:
            encoder = encoder.cpu()
        for decoder in self.decoders:
            decoder = decoder.cpu()
    
    def to_cuda(self, device):
        """将模型转移到指定GPU"""
        self.device = device
        torch.cuda.set_device(self.device)
        for encoder in self.encoders:
            encoder = encoder.cuda()
        for decoder in self.decoders:
            decoder = decoder.cuda()
