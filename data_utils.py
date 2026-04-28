# data_utils.py
# 专门用于加载传感器CSV数据的工具函数

import os
import numpy as np
import pandas as pd
from glob import glob


def load_sensor_data_from_csv(data_dir, window_size=1024, step_size=None):
    """从CSV文件加载多传感器数据

    Args:
        data_dir: 数据目录，包含多个传感器的CSV文件
        window_size: 滑动窗口大小
        step_size: 滑动步长，默认为window_size//2

    Returns:
        windows: (num_windows, num_sensors, window_size) 数据数组
        sensor_names: 传感器名称列表
    """
    if step_size is None:
        step_size = window_size // 2

    # 查找所有CSV文件
    csv_files = glob(os.path.join(data_dir, "*.csv"))
    csv_files.sort()  # 确保顺序一致

    sensor_data = {}
    sensor_names = []

    for file_path in csv_files:
        # 提取传感器名称
        sensor_name = os.path.basename(file_path).replace('.csv', '')
        sensor_names.append(sensor_name)

        # 加载CSV数据
        df = pd.read_csv(file_path, sep=';', decimal='.')

        # 提取传感器通道数据
        sensor_cols = [col for col in df.columns if col.startswith('CH')]
        data = df[sensor_cols].values.astype(np.float32)

        # 如果有多个通道，取平均值作为该传感器的值
        if data.shape[1] > 1:
            data = np.mean(data, axis=1)
        else:
            data = data.flatten()

        sensor_data[sensor_name] = data

    # 确保所有传感器数据长度一致
    min_length = min(len(data) for data in sensor_data.values())
    for name in sensor_data:
        sensor_data[name] = sensor_data[name][:min_length]

    # 构建多传感器数据矩阵 (T, N)
    num_sensors = len(sensor_names)
    T = min_length
    data_matrix = np.zeros((T, num_sensors), dtype=np.float32)

    for i, name in enumerate(sensor_names):
        data_matrix[:, i] = sensor_data[name]

    # 滑动窗口切分
    num_windows = (T - window_size) // step_size + 1
    windows = np.zeros((num_windows, num_sensors, window_size), dtype=np.float32)

    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        # (window_size, num_sensors) -> (num_sensors, window_size)
        windows[i] = data_matrix[start:end, :].T

    return windows, sensor_names


def load_labeled_dataset(data_root, window_size=1024, step_size=None):
    """加载带标签的数据集

    目录结构:
        data_root/
            condition_1/
                fault_type_1/
                    sensor_1.csv
                    sensor_2.csv
                    ...
                fault_type_2/
                    ...
            condition_2/
                ...

    Returns:
        X: (num_samples, num_sensors, window_size)
        y: (num_samples,) 故障类型标签
        conditions: (num_samples,) 工况标签
    """
    X_list = []
    y_list = []
    condition_list = []

    condition_dirs = glob(os.path.join(data_root, "condition_*"))
    condition_dirs.sort()

    for cond_idx, cond_dir in enumerate(condition_dirs):
        fault_dirs = glob(os.path.join(cond_dir, "*"))
        fault_dirs.sort()

        for fault_idx, fault_dir in enumerate(fault_dirs):
            # 加载该故障类型下的所有传感器数据
            windows, sensor_names = load_sensor_data_from_csv(
                fault_dir, window_size, step_size
            )

            for w in windows:
                X_list.append(w)
                y_list.append(fault_idx)
                condition_list.append(cond_idx)

    X = np.array(X_list)
    y = np.array(y_list)
    conditions = np.array(condition_list)

    print(f"Loaded dataset: {X.shape[0]} samples, "
          f"{X.shape[1]} sensors, "
          f"{X.shape[2]} time steps")
    print(f"Fault types: {len(np.unique(y))}, "
          f"Conditions: {len(np.unique(conditions))}")

    return X, y, conditions, sensor_names


def build_adjacency_from_positions(sensor_names, sensor_positions,
                                   correlation_matrix=None, threshold=0.6):
    """根据物理位置和/或信号相关性构建邻接矩阵"""
    N = len(sensor_names)
    adj = np.eye(N)

    # 物理邻近性
    if sensor_positions is not None:
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(sensor_positions[i] - sensor_positions[j])
                if dist < 0.5:  # 距离阈值
                    adj[i, j] = adj[j, i] = 1

    # 信号相关性
    if correlation_matrix is not None:
        mask = correlation_matrix > threshold
        adj = np.maximum(adj, mask.astype(float))

    return adj


def create_few_shot_tasks(X, y, conditions, n_ways=5, k_shots=5,
                          n_queries=10, n_tasks=100):
    """创建少样本元学习任务"""
    tasks = []

    for _ in range(n_tasks):
        # 随机选择n_ways个故障类别
        unique_labels = np.unique(y)
        if len(unique_labels) < n_ways:
            continue

        selected_labels = np.random.choice(unique_labels, n_ways, replace=False)

        support_set = []
        query_set = []

        for label in selected_labels:
            # 获取该类别的所有样本索引
            label_indices = np.where(y == label)[0]
            np.random.shuffle(label_indices)

            # 支撑集
            support_indices = label_indices[:k_shots]
            # 查询集
            query_indices = label_indices[k_shots:k_shots + n_queries]

            for idx in support_indices:
                support_set.append((X[idx], label, conditions[idx]))
            for idx in query_indices:
                query_set.append((X[idx], label, conditions[idx]))

        tasks.append((support_set, query_set))

    return tasks


def compute_correlation_matrix(X):
    """计算传感器间的相关系数矩阵

    Args:
        X: (num_samples, num_sensors, window_size)

    Returns:
        corr_matrix: (num_sensors, num_sensors)
    """
    # 对每个传感器，将所有样本拼接成一个长序列
    num_sensors = X.shape[1]
    concatenated = []

    for i in range(num_sensors):
        # (num_samples, window_size) -> (num_samples * window_size)
        sensor_seq = X[:, i, :].flatten()
        concatenated.append(sensor_seq)

    concatenated = np.array(concatenated)  # (num_sensors, total_length)
    corr_matrix = np.corrcoef(concatenated)

    # 处理NaN值
    corr_matrix = np.nan_to_num(corr_matrix)

    return corr_matrix


if __name__ == '__main__':
    # 测试数据加载
    import sys

    if len(sys.argv) > 1:
        data_root = sys.argv[1]
        X, y, conditions, sensor_names = load_labeled_dataset(data_root)
        print(f"Data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Conditions shape: {conditions.shape}")
        print(f"Sensors: {sensor_names}")

        # 计算相关系数矩阵
        corr = compute_correlation_matrix(X)
        print(f"Correlation matrix shape: {corr.shape}")

        # 构建邻接矩阵
        adj = build_adjacency_from_positions(sensor_names, None, corr)
        print(f"Adjacency matrix:\n{adj[:5, :5]}")
    else:
        print("Usage: python data_utils.py <data_root>")