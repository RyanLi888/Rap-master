"""
MADE 绘图工具模块
================

本文件提供了MADE模型训练过程中的可视化功能，包括：
1. sample_digits_maf: 从MAF模型采样并可视化数字图像
2. plot_losses: 绘制训练和验证损失曲线

作者: RAPIER 开发团队
版本: 1.0
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.distributions import MultivariateNormal


def sample_digits_maf(model, epoch, random_order=False, seed=None, test=False):
    """
    从MAF模型采样数字图像并保存
    
    参数:
        model: 训练好的MAF模型
        epoch (int): 当前训练轮次
        random_order (bool): 是否使用随机像素顺序
        seed (int): 随机种子
        test (bool): 是否为测试模式（影响保存路径）
    """
    model.eval()
    n_samples = 80
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    if random_order is True:
        np.random.seed(seed)
        order = np.random.permutation(784)
    else:
        order = np.arange(784)

    # 从标准正态分布采样隐变量
    u = torch.zeros(n_samples, 784).normal_(0, 1)
    mvn = MultivariateNormal(torch.zeros(28 * 28), torch.eye(28 * 28))
    log_prob = mvn.log_prob(u)
    
    # 通过MAF模型生成样本
    samples, log_det = model.backward(u)

    # 注释掉的代码：原实现中的排序逻辑
    # log_det = log_prob - log_det
    # log_det = log_det[np.logical_not(np.isnan(log_det.detach().numpy()))]
    # idx = np.argsort(log_det.detach().numpy())
    # samples = samples[idx].flip(dims=(0,))
    # samples = samples[80 : 80 + n_samples]

    # 将logits转换为像素值（0-1范围）
    samples = (torch.sigmoid(samples) - 1e-6) / (1 - 2e-6)
    samples = samples.detach().cpu().view(n_samples, 28, 28)

    # 创建8x10的图像网格
    fig, axes = plt.subplots(ncols=10, nrows=8)
    ax = axes.ravel()
    for i in range(n_samples):
        ax[i].imshow(
            np.transpose(samples[i], (0, 1)), cmap="gray", interpolation="none"
        )
        ax[i].axis("off")
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        ax[i].set_frame_on(False)

    # 创建保存目录并保存图像
    if test is False:
        if not os.path.exists("gif_results"):
            os.makedirs("gif_results")
        save_path = "gif_results/samples_gaussian_" + str(epoch) + ".png"
    else:
        if not os.path.exists("figs"):
            os.makedirs("figs")
        save_path = "figs/samples_gaussian_" + str(epoch) + ".png"

    fig.subplots_adjust(wspace=-0.35, hspace=0.065)
    plt.gca().set_axis_off()
    plt.savefig(
        save_path, dpi=300, bbox_inches="tight", pad_inches=0,
    )
    plt.close()


def plot_losses(epochs, train_losses, val_losses, title=None):
    """
    绘制训练和验证损失曲线
    
    参数:
        epochs (list): 训练轮次列表
        train_losses (list): 训练损失列表
        val_losses (list): 验证损失列表
        title (str, optional): 图表标题
    """
    sns.set(style="white")
    fig, axes = plt.subplots(
        ncols=1, nrows=1, figsize=[10, 5], sharey=True, sharex=True, dpi=400
    )

    # 转换为pandas Series以便绘图
    train = pd.Series(train_losses).astype(float)
    val = pd.Series(val_losses).astype(float)
    train.index += 1
    val.index += 1

    # 绘制损失曲线
    axes = sns.lineplot(data=train, color="gray", label="Training loss")
    axes = sns.lineplot(data=val, color="orange", label="Validation loss")

    # 设置图表属性
    axes.set_ylabel("Negative log-likelihood")
    axes.legend(
        frameon=False,
        prop={"size": 14},
        fancybox=False,
        handletextpad=0.5,
        handlelength=1,
    )
    axes.set_ylim(1250, 1600)
    axes.set_xlim(0, 50)
    axes.set_title(title) if title is not None else axes.set_title(None)
    
    # 创建保存目录并保存图表
    if not os.path.exists("plots"):
        os.makedirs("plots")
    save_path = "plots/train_plots" + str(epochs[-1]) + ".pdf"
    plt.savefig(
        save_path, dpi=300, bbox_inches="tight", pad_inches=0,
    )
    plt.close()
