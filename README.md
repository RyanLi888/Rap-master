# RAPIER

**论文**: Low-Quality Training Data Only? A Robust Framework for Detecting Encrypted Malicious Network Traffic

**中文说明**: 仅使用低质量训练数据？一个用于检测加密恶意网络流量的鲁棒框架

## 预处理 (Preprocessing):

```
cd ./Preprocess
python Feature_Extract.py "input_dir" "sequence_data_path" "ext"
python get_origin_flow_data.py "sequence_data_path" "save_dir" "data_type"
```

**参数说明**:
* `input_dir`: 某类数据的所有原始pcap文件目录，例如 `benign`、`malicious` 或 `test`
* `sequence_data_path`: pcap文件中所有流的序列数据，无零填充（值为前缀累积值，将由 `get_origin_flow_data.py` 进一步处理）
* `ext`: 要处理的pcap文件扩展名（例如 `pcap`、`pcapng`）
* `save_dir`: 处理后所有流序列数据的保存目录
* `data_type`: 处理后数据的类型，例如 `w`、`b` 和 `test`

**输出**: 在 `save_dir` 中生成 `data_type` 的序列numpy文件，即 `{save_dir}/{data_type}.npy`，每个样本的维度为50。需要添加第51维用于检测。

## 检测/预测 (Detection/Prediction):

```
cd ./main
python main.py
```

**可在 `main.py` 中修改的参数**:
* `data_dir`: 所有序列数据的目录
* `feat_dir`: 所有特征数据的目录
* `made_dir`: 由 `MADE` 计算的所有结果的目录
* `model_dir`: 所有训练模型的目录
* `result_dir`: 测试数据检测/预测结果的目录

**必需的输入文件**:
* `{data_dir}/{benign.npy}`: 良性预处理训练数据
* `{data_dir}/{malicious.npy}`: 恶意预处理训练数据
* `{data_dir}/{test.npy}`: 预处理测试数据

`{data_dir}` 中的所有数据应具有维度 (*n*, 51)，其中 *n* 是样本数量。每个样本是一个51维向量，前50维是流量的时间序列数据，最后一维是样本的真实标签（用于评估，`0` 表示良性，`1` 表示恶意）。如果RAPIER用于预测，最后一维可以是任意值。

**输出**:
* `{result_dir}/{prediction.npy}`: 所有测试数据的预测结果，`1` 表示恶意，`0` 表示良性

## 系统架构说明

RAPIER系统包含以下主要模块：

1. **预处理模块 (Preprocess)**: 从PCAP文件提取网络流量特征
2. **自编码器模块 (AE)**: 使用LSTM自编码器提取高维特征表示
3. **多尺度对抗判别器模块 (MADE)**: 基于密度估计的对抗样本生成
4. **分类器模块 (Classifier)**: 最终的恶意流量分类预测
5. **主控制模块 (main)**: 协调整个系统的工作流程

该系统特别适用于训练数据质量不高的情况，通过对抗样本生成和数据清理技术提高模型的鲁棒性。
