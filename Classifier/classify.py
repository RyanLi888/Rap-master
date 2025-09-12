"""
分类器训练与预测模块
====================

本文件实现了最终的分类器训练与预测流程，包括：
- 准确率计算
- Co-teaching 训练循环（两个MLP互教）
- 对测试集进行预测并保存结果
- 评估指标计算与模型保存

作者: RAPIER 开发团队
版本: 1.0
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F

from .model import MLP
import sys, os
import numpy as np
from tqdm import tqdm

from .loss import loss_coteaching

# 导入随机种子控制模块
sys.path.append('../utils')
try:
    from random_seed import deterministic_shuffle, create_deterministic_dataloader, RANDOM_CONFIG
    SEED_CONTROL_AVAILABLE = True
except ImportError:
    SEED_CONTROL_AVAILABLE = False

# 超参数
batch_size = 128
learning_rate = 1e-3
epochs = 100
num_gradual = 10
forget_rate = 0.1
exponent = 1
rate_schedule = np.ones(epochs) * forget_rate
rate_schedule[:num_gradual] = np.linspace(0, forget_rate ** exponent, num_gradual)

def accuracy(logit, target):
    """
    计算Top-1准确率
    
    参数:
        logit (Tensor): 模型原始输出 (N, C)
        target (Tensor): 真实标签 (N,)
    返回:
        Tensor: 百分比形式的准确率
    """
    output = F.softmax(logit, dim=1)
    batch_size = target.size(0)

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(100.0 / batch_size)

# 训练模型（Co-teaching）
def train(train_loader, epoch, model1, optimizer1, model2, optimizer2, device):
    """
    使用Co-teaching策略训练两个模型
    
    两个模型互相选择对方的“小损失”样本进行更新，以降低噪声标签的影响。
    
    参数:
        train_loader: 训练数据加载器
        epoch (int): 当前轮次
        model1, model2: 两个MLP模型
        optimizer1, optimizer2: 对应优化器
        device (int|None): CUDA设备ID或None
    返回:
        tuple(float, float): 两个模型各自的训练准确率
    """
    
    train_total1=0
    train_correct1=0 
    train_total2=0
    train_correct2=0 

    for i, data_labels in enumerate(train_loader):
        # 拆分特征与标签
        feats = data_labels[:, :-1].to(dtype=torch.float32)
        labels = data_labels[:, -1].to(dtype=int)
        if device != None:
            torch.cuda.set_device(device)
            feats = feats.cuda()
            labels = labels.cuda()
    
        # 前向传播与准确率
        logits1 = model1(feats)
        prec1 = accuracy(logits1, labels)
        train_total1 += 1
        train_correct1 += prec1

        logits2 = model2(feats)
        prec2 = accuracy(logits2, labels)
        train_total2 += 1
        train_correct2 += prec2
        
        # Co-teaching 损失（互选小损失样本）
        loss_1, loss_2 = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch])

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()

    train_acc1=float(train_correct1)/float(train_total1)
    train_acc2=float(train_correct2)/float(train_total2)
    return train_acc1, train_acc2

# 预测未知流量数据的标签
def predict(test_loader, model, device, alpha=0.5):
    """
    用单个模型对测试集进行预测
    
    参数:
        test_loader: 测试数据加载器
        model: 训练好的MLP模型
        device (int|None): CUDA设备ID或None
        alpha (float): 将恶意概率阈值化为二分类标签的阈值
    返回:
        np.ndarray: 二值预测 (N,)
    """
    preds = []
    for i, data in enumerate(test_loader):
        # 前向传播
        feats = data.to(dtype=torch.float32)
        
        if device != None:
            torch.cuda.set_device(device)
            feats = feats.cuda()
        
        logits = model(feats)
        outputs = F.softmax(logits, dim=1)
        preds.append((outputs[:, 1] > alpha).detach().cpu().numpy())

    return np.concatenate(preds, axis=0)

def main(feat_dir, model_dir, result_dir, TRAIN, cuda_device, parallel=5):
    """
    分类器训练与预测主流程
    
    1) 读取原始与合成特征并合并
    2) 使用Co-teaching训练两个MLP
    3) 用模型1在测试集上预测并保存
    4) 计算并保存评估指标与模型
    
    参数:
        feat_dir (str): 特征目录
        model_dir (str): 模型保存目录
        result_dir (str): 结果保存目录
        TRAIN (str): 训练数据标签前缀
        cuda_device (int|str): CUDA设备ID
        parallel (int): 并行生成的份数，用于拼接生成数据
    """
    
    cuda_device = int(cuda_device)
    # 读取原始训练集（仅前32维特征）
    be = np.load(os.path.join(feat_dir, 'be_corrected.npy'))[:, :32]
    ma = np.load(os.path.join(feat_dir, 'ma_corrected.npy'))[:, :32]
    be_shape = be.shape[0]
    ma_shape = ma.shape[0]

    # 拼接并随机抽取合成样本增强训练集
    for index in range(parallel):
        # 加载合成特征
        be_gen = np.load(os.path.join(feat_dir, 'be_%s_generated_GAN_%d.npy' % (TRAIN, index)))
        ma_gen1 = np.load(os.path.join(feat_dir, 'ma_%s_generated_GAN_1_%d.npy' % (TRAIN, index)))
        ma_gen2 = np.load(os.path.join(feat_dir, 'ma_%s_generated_GAN_2_%d.npy' % (TRAIN, index)))
        
        # 使用确定性打乱
        if SEED_CONTROL_AVAILABLE:
            be_gen = deterministic_shuffle(be_gen, seed=RANDOM_CONFIG['classifier_seed'] + index)
            ma_gen1 = deterministic_shuffle(ma_gen1, seed=RANDOM_CONFIG['classifier_seed'] + index + 1000)
            ma_gen2 = deterministic_shuffle(ma_gen2, seed=RANDOM_CONFIG['classifier_seed'] + index + 2000)
            if index == 0:
                print("✅ 分类器: 使用确定性数据打乱")
        else:
            np.random.shuffle(be_gen)
            np.random.shuffle(ma_gen1)
            np.random.shuffle(ma_gen2)
            if index == 0:
                print("⚠️  分类器: 使用非确定性数据打乱")
        be = np.concatenate([
            be, 
            be_gen[:be_shape // (parallel)], 
        ], axis=0)
        
        ma = np.concatenate([
            ma,
            ma_gen1[:ma_shape // (parallel) // 5],
            ma_gen2[:ma_shape // (parallel) // 5],
        ], axis=0)

    print(be.shape, ma.shape)

    # 组装训练集
    train_data = np.concatenate([be, ma], axis=0)
    train_label = np.concatenate([np.zeros(be.shape[0]), np.ones(ma.shape[0])], axis=0)
    train_dataset = np.concatenate((train_data, train_label[:, None]), axis=1)

    # 读取测试集
    test_data_label = np.load(os.path.join(feat_dir, 'test.npy'))
    test_data = test_data_label[:, :32]
    test_label = test_data_label[:, -1]

    device = int(cuda_device) if cuda_device != 'None' else None

    if device != None:
        torch.cuda.set_device(device)
    # 数据加载器
    print('loading dataset...')
    if SEED_CONTROL_AVAILABLE:
        train_loader = create_deterministic_dataloader(train_dataset, batch_size, shuffle=True, seed=RANDOM_CONFIG['classifier_seed'])
        print("✅ 分类器: 使用确定性数据加载器")
    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        print("⚠️  分类器: 使用非确定性数据加载器")
    # 定义两个MLP模型与优化器
    print('building model...')
    mlp1 = MLP(input_size=32, hiddens=[16, 8], output_size=2, device=device)
    if device != None:
        mlp1.to_cuda(device)
        mlp1 = mlp1.cuda()
    optimizer1 = torch.optim.Adam(mlp1.parameters(), lr=learning_rate)
    
    mlp2 = MLP(input_size=32, hiddens=[16, 8], output_size=2, device=device)
    if device != None:
        mlp2.to_cuda(device)
        mlp2 = mlp2.cuda()
    optimizer2 = torch.optim.Adam(mlp2.parameters(), lr=learning_rate)

    # 训练
    epoch=0
    mlp1.train()
    mlp2.train()
    for epoch in tqdm(range(epochs)):
        train(train_loader, epoch, mlp1, optimizer1, mlp2, optimizer2, device)
    
    # 测试与保存预测
    mlp1.eval()
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    preds = predict(test_loader, mlp1, device)
    np.save(os.path.join(result_dir, 'prediction.npy'), preds)

    # 计算评估指标
    scores = np.zeros((2, 2))
    for label, pred in zip(test_label, preds):
        scores[int(label), int(pred)] += 1
    TP = scores[1, 1]
    FP = scores[0, 1]
    TN = scores[0, 0]
    FN = scores[1, 0]
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    F1score = 2 * Recall * Precision / (Recall + Precision)
    print(Recall, Precision, F1score)
    
    with open('../data/result/detection_result.txt', 'w') as fp:
        fp.write('Testing data: Benign/Malicious = %d/%d\n'%((TN + FP), (TP + FN)))
        fp.write('Recall: %.2f, Precision: %.2f, F1: %.2f\n'%(Recall, Precision, F1score))
        fp.write('Acc: %.2f\n'%(Accuracy))

    # 保存模型
    mlp1 = mlp1.cpu()
    mlp1.to_cpu()
    torch.save(mlp1, os.path.join(model_dir, 'Detection_Model.pkl'))

def predict_only(feat_dir, model_dir, result_dir, TRAIN, cuda_device, parallel=5):
    """
    仅进行预测的分类器函数（不重新训练）
    
    该函数加载已训练好的模型，直接在测试集上进行预测，适用于使用最佳模型进行最终预测的场景。
    
    参数:
        feat_dir (str): 特征目录
        model_dir (str): 模型保存目录（包含 Detection_Model.pkl）
        result_dir (str): 结果保存目录
        TRAIN (str): 训练数据标签前缀（虽然不训练，但用于确定数据格式）
        cuda_device (int|str): CUDA设备ID
        parallel (int): 并行生成的份数（用于数据格式兼容）
    """
    
    cuda_device = int(cuda_device)
    device = int(cuda_device) if cuda_device != 'None' else None
    
    # 加载已训练的模型
    model_path = os.path.join(model_dir, 'Detection_Model.pkl')
    if not os.path.exists(model_path):
        print(f"❌ 错误：未找到已训练的模型文件 {model_path}")
        return
    
    print(f"📂 加载已训练的分类器模型: {model_path}")
    mlp1 = torch.load(model_path)
    
    if device != None:
        torch.cuda.set_device(device)
        mlp1 = mlp1.cuda()
    
    # 读取测试集
    test_data_label = np.load(os.path.join(feat_dir, 'test.npy'))
    test_data = test_data_label[:, :32]
    test_label = test_data_label[:, -1]
    
    print(f"🔍 对测试集进行预测 (测试样本数: {len(test_data)})")
    
    # 测试与保存预测
    mlp1.eval()
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    preds = predict(test_loader, mlp1, device)
    
    # 保存预测结果
    prediction_file = os.path.join(result_dir, 'prediction.npy')
    np.save(prediction_file, preds)
    print(f"💾 预测结果已保存到: {prediction_file}")
    
    # 计算评估指标
    scores = np.zeros((2, 2))
    for label, pred in zip(test_label, preds):
        scores[int(label), int(pred)] += 1
    TP = scores[1, 1]
    FP = scores[0, 1]
    TN = scores[0, 0]
    FN = scores[1, 0]
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    F1score = 2 * Recall * Precision / (Recall + Precision) if (Recall + Precision) > 0 else 0.0
    
    print(f"📊 最终预测结果:")
    print(f"   准确率: {Accuracy:.4f}")
    print(f"   召回率: {Recall:.4f}")  
    print(f"   精确率: {Precision:.4f}")
    print(f"   F1分数: {F1score:.4f}")
    
    # 保存结果到文件
    result_file = os.path.join(result_dir, 'final_detection_result.txt')
    with open(result_file, 'w') as fp:
        fp.write('=== 最终预测结果（使用最佳模型） ===\n')
        fp.write('Testing data: Benign/Malicious = %d/%d\n'%((TN + FP), (TP + FN)))
        fp.write('Recall: %.4f, Precision: %.4f, F1: %.4f\n'%(Recall, Precision, F1score))
        fp.write('Accuracy: %.4f\n'%(Accuracy))
    
    print(f"📄 详细结果已保存到: {result_file}")
    
    return F1score

def predict_only_from_file(feat_dir, model_file_path, result_dir, TRAIN, cuda_device, parallel=5):
    """
    从指定的模型文件进行预测（不重新训练）
    
    该函数从指定的模型文件加载模型，直接在测试集上进行预测。
    专门用于加载最佳模型文件进行最终预测。
    
    参数:
        feat_dir (str): 特征目录
        model_file_path (str): 模型文件的完整路径
        result_dir (str): 结果保存目录
        TRAIN (str): 训练数据标签前缀
        cuda_device (int|str): CUDA设备ID
        parallel (int): 并行生成的份数
        
    返回:
        float: F1分数
    """
    
    cuda_device = int(cuda_device)
    device = int(cuda_device) if cuda_device != 'None' else None
    
    # 检查模型文件是否存在
    if not os.path.exists(model_file_path):
        print(f"❌ 错误：模型文件不存在 {model_file_path}")
        return 0.0
    
    print(f"📂 从指定文件加载最佳分类器模型: {model_file_path}")
    
    try:
        # 加载模型
        mlp1 = torch.load(model_file_path)
        
        if device != None:
            torch.cuda.set_device(device)
            mlp1 = mlp1.cuda()
        
        # 读取测试集
        test_data_label = np.load(os.path.join(feat_dir, 'test.npy'))
        test_data = test_data_label[:, :32]
        test_label = test_data_label[:, -1]
        
        print(f"🔍 使用最佳模型对测试集进行预测 (测试样本数: {len(test_data)})")
        
        # 测试与保存预测
        mlp1.eval()
        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
        preds = predict(test_loader, mlp1, device)
        
        # 保存预测结果
        prediction_file = os.path.join(result_dir, 'best_model_prediction.npy')
        np.save(prediction_file, preds)
        print(f"💾 最佳模型预测结果已保存到: {prediction_file}")
        
        # 计算评估指标
        scores = np.zeros((2, 2))
        for label, pred in zip(test_label, preds):
            scores[int(label), int(pred)] += 1
        TP = scores[1, 1]
        FP = scores[0, 1]
        TN = scores[0, 0]
        FN = scores[1, 0]
        
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        Recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        F1score = 2 * Recall * Precision / (Recall + Precision) if (Recall + Precision) > 0 else 0.0
        
        print(f"📊 最佳模型预测结果:")
        print(f"   准确率: {Accuracy:.4f}")
        print(f"   召回率: {Recall:.4f}")  
        print(f"   精确率: {Precision:.4f}")
        print(f"   F1分数: {F1score:.4f}")
        
        # 保存结果到文件
        result_file = os.path.join(result_dir, 'best_model_final_result.txt')
        with open(result_file, 'w') as fp:
            fp.write('=== 最佳模型最终预测结果 ===\n')
            fp.write(f'使用的模型文件: {model_file_path}\n')
            fp.write('Testing data: Benign/Malicious = %d/%d\n'%((TN + FP), (TP + FN)))
            fp.write('Recall: %.4f, Precision: %.4f, F1: %.4f\n'%(Recall, Precision, F1score))
            fp.write('Accuracy: %.4f\n'%(Accuracy))
        
        print(f"📄 最佳模型详细结果已保存到: {result_file}")
        
        return F1score
        
    except Exception as e:
        print(f"❌ 从文件加载模型时出错: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

