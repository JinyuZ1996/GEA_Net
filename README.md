# GEA-Net PyTorch

This project provides a PyTorch implementation of **GEA-Net (Graph External Attention Network)** proposed in the paper *GEA-Net: A Lightweight Graph External Attention Network for Industrial Equipment Fault Diagnosis*. The framework supports both the **MCC5-THU** and **SEU** datasets through a unified pipeline. It includes automatic dataset discovery, adaptive data formatting, leakage-free sliding-window splitting, preprocessing cache, graph construction based on correlation and physical priors, GEA-Net, standard supervised training, Proto-MAML few-shot learning, target-domain prototype evaluation, performance metrics, and testing.

## Environment

Recommended environment:

-   Python 3.10--3.12
-   PyTorch 2.1

Install the dependencies with:

``` powershell
python -m pip install -r requirements.txt
```

## Training

### Cross-Condition Few-Shot Learning (Main Experimental Pipeline)

``` powershell
python train.py --config configs/seu.yaml --mode fewshot
python train.py --config configs/mcc5_thu.yaml --mode fewshot
```

For the SEU dataset, **1200 RPM** and **1800 RPM** are used as the
source conditions by default, while **2400 RPM** is used as the target
condition. For MCC5-THU, the last one-third of the sorted operating
conditions are reserved as the target domain by default. After the
dataset is finalized, it is recommended to explicitly specify the four
target conditions used in the paper through the `target_conditions`
field.

### Standard Supervised Training

``` powershell
python train.py --config configs/seu.yaml --mode supervised
```

In supervised mode, only the training split of the source conditions is
used for model training. Model selection is performed on the validation
split of the source conditions, and zero-shot cross-condition
performance is evaluated on the test split of the target conditions. For
random split or same-condition supervised experiments, simply set
`target_conditions` to an empty list (`[]`).

### Evaluate a Checkpoint

``` powershell
python evaluate.py --checkpoint runs\seu\best.pt
python evaluate.py --checkpoint runs\seu\best.pt --episodes 500
```

## Data Splitting and Leakage Prevention

Each raw recording is first divided into non-overlapping temporal
windows. The windows are then split sequentially into **70% training**,
**15% validation**, and **15% testing** within the same recording. This
strategy avoids data leakage caused by randomly shuffling neighboring
windows.

In few-shot mode:

-   Source-condition **train**: support/query sets for Proto-MAML
    meta-training.
-   Source-condition **train → validation**: model selection.
-   Target-condition **train → test**: K-shot support set and
    independent query set.
-   During target-domain evaluation, only class prototypes are computed;
    no gradient updates are performed.

`normalization: per_window` independently normalizes each sensor in
every window. If signal amplitude itself is an important diagnostic
feature, `per_recording` or `none` can be used instead.

## Unified Interface for Both Datasets

The model accepts inputs in the following format:

``` text
x: [batch, num_sensors, window_size]
```

The number of sensors is determined automatically from the cached
metadata, while the number of fault classes is inferred through dataset
discovery. Therefore, the same model implementation supports both the
**3-sensor SEU** dataset and the **6-sensor MCC5-THU** dataset by simply
switching the YAML configuration.

The following options can be configured in the YAML file:

-   `window_size` / `stride`: sliding-window configuration.
-   `include_conditions` / `target_conditions`: source and target
    operating conditions.
-   `include_labels`: selected fault categories.
-   `channel_names`: explicitly specify signal channels.
-   `embed_dim` / `memory_size` / `graph_layers`: model architecture.
-   `graph_beta`: fusion weight between structural priors and dynamic
    attention.
-   `ways` / `shots` / `queries`: few-shot episode settings.
-   `first_order`: first-order or second-order MAML.

## Outputs

Each training run generates:

-   `best.pt`: best checkpoint on the validation set.
-   `last.pt`: final checkpoint with training summary.
-   `history.json`: metrics recorded for each epoch.
-   `summary.json`: target-domain test results, model size, and runtime
    statistics.
-   `evaluation.json`: results produced by `evaluate.py`.

The preprocessing cache is indexed using the original file path, file
size, modification timestamp, and sliding-window configuration. Whenever
the raw data or key preprocessing settings change, a new cache is
created automatically.
