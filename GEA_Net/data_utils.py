# data_utils.py

import os
import random
import numpy as np
import pandas as pd
from glob import glob


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_sensor_csv(file_path):
    df = pd.read_csv(file_path, sep=';', decimal='.')
    sensor_cols = [col for col in df.columns if col.startswith('CH')]
    data = df[sensor_cols].values.astype(np.float32)
    if data.shape[1] > 1:
        data = np.mean(data, axis=1)
    else:
        data = data.flatten()
    return data


def sliding_window(data, window_size, step_size=None):
    if step_size is None:
        step_size = window_size // 2
    num_samples = (data.shape[0] - window_size) // step_size + 1
    windows = []
    for i in range(num_samples):
        start = i * step_size
        end = start + window_size
        windows.append(data[start:end])
    return np.array(windows)


def load_labeled_dataset(data_root, window_size=1024):
    X_list = []
    y_list = []
    condition_list = []
    sensor_names = []

    condition_dirs = glob(os.path.join(data_root, "condition_*"))
    condition_dirs.sort()

    for cond_idx, cond_dir in enumerate(condition_dirs):
        fault_dirs = glob(os.path.join(cond_dir, "*"))
        fault_dirs.sort()

        for fault_idx, fault_dir in enumerate(fault_dirs):
            csv_files = glob(os.path.join(fault_dir, "*.csv"))
            csv_files.sort()

            if len(sensor_names) == 0:
                for f in csv_files:
                    name = os.path.basename(f).replace('.csv', '')
                    sensor_names.append(name)

            sensor_data_list = []
            for csv_file in csv_files:
                data = load_sensor_csv(csv_file)
                sensor_data_list.append(data)

            min_length = min(len(d) for d in sensor_data_list)
            for i in range(len(sensor_data_list)):
                sensor_data_list[i] = sensor_data_list[i][:min_length]

            num_sensors = len(sensor_data_list)
            data_matrix = np.zeros((min_length, num_sensors), dtype=np.float32)
            for i in range(num_sensors):
                data_matrix[:, i] = sensor_data_list[i]

            windows = sliding_window(data_matrix, window_size)
            windows = windows.transpose(0, 2, 1)

            for w in windows:
                X_list.append(w)
                y_list.append(fault_idx)
                condition_list.append(cond_idx)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    conditions = np.array(condition_list)

    print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} sensors, {X.shape[2]} time steps")
    print(f"Fault types: {len(np.unique(y))}, Conditions: {len(np.unique(conditions))}")

    return X, y, conditions, sensor_names


def compute_correlation_matrix(X):
    num_sensors = X.shape[1]
    concatenated = []
    for i in range(num_sensors):
        sensor_seq = X[:, i, :].flatten()
        concatenated.append(sensor_seq)
    concatenated = np.array(concatenated)
    corr_matrix = np.corrcoef(concatenated)
    corr_matrix = np.nan_to_num(corr_matrix)
    return corr_matrix


def build_initial_adjacency(sensor_names, correlation_matrix, threshold=0.6):
    N = len(sensor_names)
    adj = np.eye(N)
    if correlation_matrix is not None:
        mask = correlation_matrix > threshold
        adj = np.maximum(adj, mask.astype(float))
    for i in range(N - 1):
        adj[i, i + 1] = adj[i + 1, i] = max(adj[i, i + 1], 0.5)
    return adj


def normalize_adjacency(adj):
    adj = adj + np.eye(adj.shape[0])
    d = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adj_norm = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    return adj_norm.astype(np.float32)


def create_few_shot_tasks(X, y, conditions, n_ways=5, k_shots=5, n_queries=10, n_tasks=100):
    tasks = []
    unique_labels = np.unique(y)
    if len(unique_labels) < n_ways:
        n_ways = len(unique_labels)
    for _ in range(n_tasks):
        selected_labels = np.random.choice(unique_labels, n_ways, replace=False)
        support_set = []
        query_set = []
        for label in selected_labels:
            label_indices = np.where(y == label)[0]
            if len(label_indices) < k_shots + n_queries:
                continue
            np.random.shuffle(label_indices)
            support_indices = label_indices[:k_shots]
            query_indices = label_indices[k_shots:k_shots + n_queries]
            for idx in support_indices:
                support_set.append((X[idx], label, conditions[idx]))
            for idx in query_indices:
                query_set.append((X[idx], label, conditions[idx]))
        if len(support_set) >= n_ways * k_shots and len(query_set) >= n_ways * n_queries:
            tasks.append((support_set, query_set))
    return tasks


def normalize_sequence(X):
    X_norm = X.copy()
    num_samples, num_sensors, window_size = X.shape
    for i in range(num_samples):
        for j in range(num_sensors):
            mean = np.mean(X[i, j, :])
            std = np.std(X[i, j, :])
            if std > 1e-6:
                X_norm[i, j, :] = (X[i, j, :] - mean) / std
    return X_norm


def generate_batches(X, y, batch_size, is_train=True):
    num_samples = len(X)
    indices = np.arange(num_samples)
    if is_train:
        np.random.shuffle(indices)
    num_batches = (num_samples + batch_size - 1) // batch_size
    batches = []
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_samples)
        batch_indices = indices[start:end]
        batch_X = X[batch_indices]
        batch_y = y[batch_indices]
        batches.append((batch_X, batch_y))
    return batches, num_batches