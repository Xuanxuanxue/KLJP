# KLJP

本项目实现论文 **Legal Judgment Prediction based on Knowledge-enhanced
Multi-Task and Multi-Label Text Classification** 中的知识增强法律判决预测模型。

论文地址：[K-LJP (NAACL 2025)](https://aclanthology.org/2025.naacl-long.355.pdf)

## 1. 运行方式

默认配置位于 `contra/setting.py`：

```python
MODEL = "Al_Trans"
ADD_DETAILS = True
```

Electra 加载采用本地优先策略。若存在本地目录，会直接读取：

```text
/mnt/e/PythonProject1/kljp2/chinese-electra-180g-small-discriminator
```

该目录对应 Windows 路径：

```text
E:\PythonProject1\kljp2\chinese-electra-180g-small-discriminator
```

如果本地目录不存在或读取失败，程序才尝试使用
`hfl/chinese-electra-180g-small-discriminator`。

训练：

```bash
conda activate kljp
python contra/train_bl.py
```

默认每 50 个 batch 记录一次日志，也可以修改记录间隔：

```bash
python contra/train_bl.py --log-interval 100
```

每次训练会生成独立目录：

```text
\\wsl.localhost\Ubuntu-20.04\home\user\KLJP-logs/<model_name>/<run_id>/
├── config.json
├── train.log
├── test.log
├── metrics.jsonl
├── checkpoints/
└── tensorboard/
```

训练输出统一写入：

```text
/home/user/KLJP-logs/
└── <model_name>/<run_id>/
    ├── config.json
    ├── train.log
    ├── test.log
    ├── metrics.jsonl
    ├── checkpoints/
    ├── predictions/
    └── tensorboard/
```

Windows 文件管理器访问路径：

```text
\\wsl.localhost\Ubuntu-20.04\home\user\KLJP-logs
```

查看 TensorBoard：

```bash
tensorboard --logdir /home/user/KLJP-logs
```

## 2. 代码中的基线模型

`contra/setting.py` 中列出的模型由 `contra/train_bl.py` 统一构造。下表中的基线模型用于与知识增强模型比较：

| `MODEL` | 类型 | 主要结构 | 代码文件 | 当前状态 |
|---|---|---|---|---|
| `TextCNN` | 传统神经网络基线 | Embedding → 多尺度 CNN → Max Pooling → Linear | `models/TextCNN.py` | 可由 `Trainer` 直接构造 |
| `LSTM` | 序列模型基线 | Embedding → 双向 LSTM → Attention Pooling → Linear | `models/LSTM_attn.py` | 可由 `Trainer` 直接构造 |
| `Transformer` | Transformer 基线 | Embedding → Transformer Encoder → Attention Pooling → Linear | `models/transformer_attn.py` | 可由 `Trainer` 直接构造 |
| `TopJudge` | 结构化判决基线 | CNN Encoder → LSTM Decoder → Label Predictor | `models/TopJudge_tmp.py` | 可由 `Trainer` 直接构造 |
| `Electra` | 预训练模型基线 | Chinese Electra → `[CLS]` 表示 → Linear | `models/electra.py` | 可由 `Trainer` 直接构造 |
| `Attention_XML` | 多标签注意力基线 | BiLSTM → Label-aware/XML Attention → 多标签预测 | `models/attention_xml.py` | 可由 `Trainer` 直接构造 |

以下两个模型不是普通基线，而是本项目的知识增强模型变体：

| `MODEL` | 定位 | 主要增加内容 |
|---|---|---|
| `Al_Trans` | K-LJP 主模型 | Electra、法条/罪名定义、双任务 Transformer Decoder、A2C/C2A 对齐 |
| `CNN_Trans` | CNN 编码器变体 | CNN Fact Encoder、标签详情、双任务 Transformer Decoder |

### 模型选择

修改 `contra/setting.py`：

```python
MODEL = "TextCNN"       # 基线示例
# MODEL = "LSTM"
# MODEL = "Transformer"
# MODEL = "TopJudge"
# MODEL = "Electra"
# MODEL = "Attention_XML"
# MODEL = "Al_Trans"    # 默认知识增强模型
# MODEL = "CNN_Trans"   # CNN 知识增强变体
```

### 需要注意的实现状态

- `Bert` 出现在 `setting.py` 的注释和数据加载逻辑中，但 `Trainer` 当前没有 `MODEL == "Bert"` 的模型构造分支，因此不能直接通过 `python contra/train_bl.py` 运行；`models/Bert.py` 只是保留的模型文件。
- `TopJudge.py` 和 `TopJudge_tmp.py` 都存在，当前 `Trainer` 实际导入的是 `TopJudge_tmp.py`。
- 基线模型使用 raw logits，训练时采用 `BCEWithLogitsLoss`，评估阈值为 `0`。
- `Al_Trans` 和 `CNN_Trans` 的 Predictor 按论文 Equation (2) 输出 sigmoid 概率，训练时采用 `BCELoss`，评估阈值为 `0.5`。
- 当前 `Trainer` 的 `model_name` 默认包含 `al1`，因此会对所有模型执行额外的 `al_loss`。如果要进行严格的“无对齐损失”基线实验，需要在 `train_bl.py` 中关闭该分支，而不能只修改 `MODEL`。

基线模型调用关系：

```text
Trainer
└── MODEL
    ├── TextCNN       ── models/TextCNN.py
    ├── LSTM          ── models/LSTM_attn.py
    ├── Transformer   ── models/transformer_attn.py
    ├── TopJudge      ── models/TopJudge_tmp.py
    ├── Electra       ── models/electra.py
    ├── Attention_XML ── models/attention_xml.py
    ├── Al_Trans      ── models/el_trans.py
    └── CNN_Trans     ── models/cnn_trans.py
```

## 3. 主要模型结构

`contra/train_bl.py` 中的 `Trainer` 负责数据加载、模型构造、损失计算和训练。
默认使用 `Al_Trans`：

```text
Trainer
├── simple_load_multi_data
│   ├── 事实描述 fact description
│   ├── 法条标签 article labels
│   ├── 罪名标签 charge labels
│   └── A2C / C2A 标签映射
├── load_details
│   ├── 法条定义
│   └── 罪名定义
├── Al_Trans
│   ├── Electra Fact Encoder
│   │   └── 事实序列 h_f
│   ├── Label Detail Encoder
│   │   ├── 法条定义 → Q^a_0
│   │   └── 罪名定义 → Q^c_0
│   ├── Law Article Branch
│   │   └── Transformer Decoder × 2
│   │       ├── Self-Attention
│   │       ├── Cross-Attention(Q, h_f, h_f)
│   │       └── Feed-Forward Network
│   ├── Charge Branch
│   │   └── Transformer Decoder × 2
│   │       ├── Self-Attention
│   │       ├── Cross-Attention(Q, h_f, h_f)
│   │       └── Feed-Forward Network
│   ├── Law Article Predictor
│   │   └── GroupWiseLinear → sigmoid → article probabilities
│   └── Charge Predictor
│       └── GroupWiseLinear → sigmoid → charge probabilities
└── Losses
    ├── Multi-label BCE loss
    └── A2C / C2A alignment loss
```

## 4. Decoder 参数说明

当前 `Al_Trans` 的两个 Transformer Decoder 均使用以下配置：

```python
d_model=hid_dim              # 当前为 256
nhead=4                       # 4 个注意力头，每头 64 维
num_encoder_layers=1         # Transformer 内部的事实编码层
num_decoder_layers=2         # Decoder 层数
dim_feedforward=hid_dim * 8  # 当前为 2048
dropout=0.1
activation="relu"
normalize_before=False       # Post-Norm
```

为了对应论文 Figure 3，当前两个任务分支显式保留 Decoder Self-Attention：

```python
rm_self_attn_dec=False
rm_first_self_attn=False
```

这两个参数的含义是：

- `rm_self_attn_dec=True`：删除第一层之后 Decoder 层的 Self-Attention；
- `rm_first_self_attn=True`：删除第一层 Decoder 的 Self-Attention；
- 两者都为 `False`：所有 Decoder 层都保留 Self-Attention。

因此每个 Decoder Layer 的计算顺序为：

```text
Q_{i-1}
   │
   ├── Self-Attention(Q_{i-1}, Q_{i-1}, Q_{i-1})
   │
   ├── Cross-Attention(Q'_i, h_f, h_f)
   │
   └── FFN(Q''_i)
        ↓
      Q_i
```

其中 Self-Attention 学习标签之间的关系，Cross-Attention 将标签表示与事实描述关联，FFN 对每个标签位置进行独立的非线性变换。

## 5. Predictor 与论文 Equation (2)

法条和罪名预测器都使用独立的 Group-wise Linear 参数。对于第 `i` 个标签，模型计算：

```text
logit_i = W_i^T h_i
probability_i = sigmoid(logit_i)
```

这对应论文中的：

```text
y_hat_ij = sigmoid(W_ij^T h_i^*)
```

当前 Predictor 不使用 bias，并直接返回 `[0, 1]` 范围内的概率。因此 `Al_Trans` 和 `CNN_Trans` 使用 `BCELoss`，推理阈值为 `0.5`。其他仍输出 raw logits 的基线模型继续使用 `BCEWithLogitsLoss` 和阈值 `0`。

## 6. 主要创新点调用树

```text
K-LJP
├── Label-level Knowledge（标签级知识）
│   ├── 法条定义
│   ├── 罪名定义
│   ├── Transformer 编码定义文本
│   └── 得到初始标签表示 Q^a_0 / Q^c_0
│
├── Label-aware Transformer Decoder
│   ├── Self-Attention
│   │   └── 建模标签之间的依赖关系
│   ├── Cross-Attention
│   │   └── 将标签查询连接到事实描述
│   └── FFN
│       └── 生成每个标签的最终表示
│
├── Multi-Task Multi-Label Prediction
│   ├── Law Article Predictor
│   │   └── 预测多个相关法条
│   └── Charge Predictor
│       └── 预测多个相关罪名
│
└── Task-level Knowledge（任务级知识）
    ├── A2C：法条 → 罪名
    ├── C2A：罪名 → 法条
    └── Alignment Loss
        └── 约束两个任务的预测分布保持一致
```

## 7. 文件职责

```text
contra/
├── train_bl.py             # Trainer、训练循环、BCE 和对齐损失
├── dataloader2.py          # 数据集、标签映射、标签详情加载
├── transformer.py          # Encoder、Decoder、Self/Cross-Attention、FFN
├── models/el_trans.py      # Electra + 标签详情 + 双任务 Transformer
├── models/cnn_trans.py     # CNN 编码器版本
├── position_encoding.py    # 位置编码
└── setting.py              # 模型和训练配置
```

## 8. 数据目录

```text
KLJP-DATA/
├── dataset/                # train.json、valid.json、test.json
├── label2id/               # 法条和罪名编号
├── label2label/            # A2C、C2A 映射
└── label_details/          # 法条和罪名定义
```

数据下载地址：

[百度网盘数据](https://pan.baidu.com/s/1gc-59wZd-vRcMdwVyoOXHg?pwd=jufh)，提取码：`jufh`

## 9. 测试

```bash
/root/miniconda3/envs/kljp/bin/python -m unittest tests/test_encoder_decoder.py -v
```

测试覆盖 Encoder/Decoder 的基本前向传播、padding mask 和反向传播可用性。
