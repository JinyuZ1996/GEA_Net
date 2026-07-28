# GEA-Net PyTorch

本项目根据《面向设备故障诊断的轻量图外部注意力网络》实现了一套可同时处理 MCC5-THU 与 SEU 的 PyTorch 代码。它包含数据发现、格式自适应、无泄漏滑窗切分、缓存、相关性/物理先验构图、GEA-Net、普通监督训练、Proto-MAML 少样本训练、目标工况原型评估、指标和测试。

## 环境

建议 Python 3.10～3.12、PyTorch 2.1 

若在其他环境中安装：

```powershell
python -m pip install -r requirements.txt
```

## 正式训练

### 跨工况少样本（论文主流程）

```powershell
python train.py --config configs/seu.yaml --mode fewshot
python train.py --config configs/mcc5_thu.yaml --mode fewshot
```

SEU 默认以 1200、1800 RPM 为源工况，2400 RPM 为目标工况。MCC5-THU 默认按排序后的工况将后 1/3 留作目标工况；建议数据到位后在 `target_conditions` 中显式写出论文实验使用的 4 个目标工况。

### 普通监督训练

```powershell
python train.py --config configs/seu.yaml --mode supervised
```

监督模式只使用源工况的训练段训练，在源工况验证段选模型，并在目标工况测试段报告零样本跨工况性能。若要做随机/同工况监督实验，把 `target_conditions` 设置为空列表 `[]`。

### 评估检查点

```powershell
python evaluate.py --checkpoint runs\seu\best.pt
python evaluate.py --checkpoint runs\seu\best.pt --episodes 500
```

## 数据切分与防泄漏

每个原始长记录先按时间顺序切成不重叠窗口，再在同一记录内按前 70% / 中间 15% / 后 15% 分为 train / val / test。这样避免随机打散相邻窗口造成明显泄漏。

少样本模式：

- 源工况 train：Proto-MAML 元训练的支撑集与查询集；
- 源工况 train → val：模型选择；
- 目标工况 train → test：K-shot 支撑集与独立查询集；
- 目标评估阶段只计算类别原型，不做梯度更新。

`normalization: per_window` 对每个窗口、每个传感器独立标准化。若幅值本身是主要诊断特征，可改为 `per_recording` 或 `none`。

## 两套数据集共用的接口

模型统一接收：

```text
x: [batch, num_sensors, window_size]
```

传感器数由缓存清单自动确定，类别数由文件发现结果自动确定。因此同一模型代码可以处理 3 节点 SEU 和 6 节点 MCC5-THU，只需切换 YAML。

可在 YAML 中控制：

- `window_size` / `stride`：滑窗；
- `include_conditions` / `target_conditions`：源目标工况；
- `include_labels`：限制故障类别；
- `channel_names`：显式指定信号列；
- `embed_dim` / `memory_size` / `graph_layers`；
- `graph_beta`：结构先验与动态注意力的融合权重；
- `ways` / `shots` / `queries`：少样本 episode；
- `first_order`：一阶或二阶 MAML。

## 输出

每次训练目录包含：

- `best.pt`：验证集最优检查点；
- `last.pt`：带最终摘要的检查点；
- `history.json`：逐 epoch 指标；
- `summary.json`：目标测试、参数量和耗时；
- `evaluation.json`：单独运行 `evaluate.py` 的结果。

预处理缓存按原文件路径、大小、修改时间和滑窗配置生成哈希目录。修改数据或关键预处理参数后会自动创建新缓存。
