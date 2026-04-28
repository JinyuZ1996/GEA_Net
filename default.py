# GEA_Net_Complete.py
# Lightweight Graph External Attention Network for Fault Diagnosis
# TensorFlow v1 Implementation

from time import time
import random
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
import logging
from glob import glob
from collections import defaultdict
from sklearn.model_selection import train_test_split

np.seterr(all='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

seed = 2025
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)


################## Part 1: 超参数配置 ##################
class ParamConfig:
    def __init__(self):
        '''
           block 1: 训练超参数
        '''
        self.learning_rate = 0.01
        self.dropout_rate = 0.1
        self.batch_size = 256
        self.num_epochs = 100
        self.eval_verbose = 10
        self.fast_running = False
        self.fast_ratio = 0.5

        '''
            block 2: 模型超参数
        '''
        self.embedding_size = 16  # d: 嵌入维度
        self.num_layers = 2  # L: 图卷积层数
        self.external_memory_dim = 16  # α: 外部记忆单元维度
        self.beta = 0.8  # β: 平衡系数
        self.lambda_1 = 0.1  # λ1: 一致性正则化系数
        self.lambda_2 = 0.01  # λ2: 图结构正则化系数
        self.window_size = 1024  # T: 时间窗口长度
        self.threshold_tau = 0.6  # τ: 相关系数阈值

        '''
            block 3: 路径配置
        '''
        self.data_dir = "./Data/sensor_data/"
        self.check_points = "./check_points/GEA_Net.ckpt"
        self.gpu_index = '0'


################## Part 2: 数据加载与预处理 ##################
def load_sensor_csv(file_path):
    """加载单个传感器CSV文件"""
    df = pd.read_csv(file_path, sep=';', decimal='.')
    sensor_cols = [col for col in df.columns if col.startswith('CH')]
    data = df[sensor_cols].values.astype(np.float32)
    # 多通道取平均
    if data.shape[1] > 1:
        data = np.mean(data, axis=1)
    else:
        data = data.flatten()
    return data


def sliding_window(data, window_size, step_size=None):
    """滑动窗口切分数据"""
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
    """加载带标签的数据集

    目录结构:
        data_root/
            condition_0/
                fault_0/
                    sensor_1.csv
                    sensor_2.csv
                    ...
                fault_1/
                    ...
            condition_1/
                ...
    """
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
            # 加载该故障类型下的所有传感器数据
            csv_files = glob(os.path.join(fault_dir, "*.csv"))
            csv_files.sort()

            if len(sensor_names) == 0:
                for f in csv_files:
                    name = os.path.basename(f).replace('.csv', '')
                    sensor_names.append(name)

            # 读取所有传感器数据
            sensor_data_list = []
            for csv_file in csv_files:
                data = load_sensor_csv(csv_file)
                sensor_data_list.append(data)

            # 确保所有传感器数据长度一致
            min_length = min(len(d) for d in sensor_data_list)
            for i in range(len(sensor_data_list)):
                sensor_data_list[i] = sensor_data_list[i][:min_length]

            # 构建多传感器矩阵 (T, N)
            num_sensors = len(sensor_data_list)
            data_matrix = np.zeros((min_length, num_sensors), dtype=np.float32)
            for i in range(num_sensors):
                data_matrix[:, i] = sensor_data_list[i]

            # 滑动窗口切分
            windows = sliding_window(data_matrix, window_size)
            # windows shape: (num_windows, window_size, num_sensors)
            # 转换为 (num_windows, num_sensors, window_size)
            windows = windows.transpose(0, 2, 1)

            for w in windows:
                X_list.append(w)
                y_list.append(fault_idx)
                condition_list.append(cond_idx)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    conditions = np.array(condition_list)

    print(f"Loaded dataset: {X.shape[0]} samples, "
          f"{X.shape[1]} sensors, "
          f"{X.shape[2]} time steps")
    print(f"Fault types: {len(np.unique(y))}, "
          f"Conditions: {len(np.unique(conditions))}")
    print(f"Sensors: {sensor_names}")

    return X, y, conditions, sensor_names


def compute_correlation_matrix(X):
    """计算传感器间的相关系数矩阵 (N, N)"""
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
    """构建初始邻接矩阵"""
    N = len(sensor_names)
    adj = np.eye(N)

    # 基于信号相关性构图
    if correlation_matrix is not None:
        mask = correlation_matrix > threshold
        adj = np.maximum(adj, mask.astype(float))

    # 添加一些物理邻近性连接（可根据实际传感器位置调整）
    # 这里假设传感器按物理顺序排列，相邻传感器有连接
    for i in range(N - 1):
        adj[i, i + 1] = adj[i + 1, i] = max(adj[i, i + 1], 0.5)

    return adj


def normalize_adjacency(adj):
    """对称归一化邻接矩阵: D^{-1/2} A D^{-1/2}"""
    adj = adj + np.eye(adj.shape[0])
    d = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adj_norm = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    return adj_norm.astype(np.float32)


def create_few_shot_tasks(X, y, conditions, n_ways=5, k_shots=5, n_queries=10, n_tasks=100):
    """创建少样本元学习任务"""
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
    """对每个传感器的序列进行Z-score归一化"""
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
    """生成批次数据"""
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


################## Part 3: GEA-Net模型 ##################
class GEA_Net:
    """轻量图外部注意力网络"""

    def __init__(self, num_sensors, num_classes, config):
        os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_index
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        self.num_sensors = num_sensors
        self.num_classes = num_classes
        self.ebd_size = config.embedding_size
        self.num_layers = config.num_layers
        self.external_mem_dim = config.external_memory_dim
        self.window_size = config.window_size
        self.beta = config.beta
        self.lambda_1 = config.lambda_1
        self.lambda_2 = config.lambda_2
        self.learning_rate = config.learning_rate
        self.dropout_rate = config.dropout_rate

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope('inputs'):
                self.build_placeholders()

            with tf.name_scope('gea_net'):
                self.build_model()

            with tf.name_scope('loss'):
                self.build_loss()

            with tf.name_scope('optimizer'):
                self.build_optimizer()

            with tf.name_scope('metrics'):
                self.build_metrics()

    def build_placeholders(self):
        """构建输入占位符"""
        # 多传感器时间序列: (batch, N_sensors, T)
        self.X = tf.placeholder(tf.float32,
                                shape=[None, self.num_sensors, self.window_size],
                                name='input_sequences')
        # 标签
        self.y = tf.placeholder(tf.int32, shape=[None], name='labels')
        # 初始邻接矩阵
        self.adj_init = tf.placeholder(tf.float32,
                                       shape=[self.num_sensors, self.num_sensors],
                                       name='adjacency_init')
        # 学习率
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        # Dropout rate
        self.dropout = tf.placeholder(tf.float32, name='dropout_rate')
        # 是否训练模式
        self.is_training = tf.placeholder(tf.bool, name='is_training')

    def external_attention(self, queries, memory_K, memory_V):
        """外部注意力机制

        Args:
            queries: (B, d) 查询
            memory_K: (α, d) 外部键记忆单元
            memory_V: (α, d) 外部值记忆单元

        Returns:
            output: (B, d) 注意力加权输出
            attention: (B, α) 注意力矩阵
        """
        # Q @ M_K^T: (B, α)
        scores = tf.matmul(queries, memory_K, transpose_b=True)
        scores = scores / tf.sqrt(tf.cast(tf.shape(queries)[-1], tf.float32))

        # L2归一化
        attention = tf.nn.l2_normalize(scores, axis=-1)

        # A @ M_V: (B, d)
        output = tf.matmul(attention, memory_V)

        return output, attention

    def temporal_encoding_branch(self, X):
        """时序编码分支

        Args:
            X: (batch, N_sensors, T) 原始时间序列

        Returns:
            F_seq: (N_sensors, d) 时序嵌入矩阵
            W_att: (N_sensors, N_sensors) 注意力得分矩阵
        """
        with tf.variable_scope('temporal_encoding', reuse=tf.AUTO_REUSE):
            batch_size = tf.shape(X)[0]
            N = self.num_sensors
            d = self.ebd_size
            alpha = self.external_mem_dim

            # 线性变换: (batch, N, T) -> (batch, N, d)
            X_reshaped = tf.reshape(X, [-1, self.window_size])  # (batch*N, T)

            W_proj = tf.get_variable('W_proj', shape=[self.window_size, d],
                                     initializer=tf.contrib.layers.xavier_initializer())
            X_embedded = tf.matmul(X_reshaped, W_proj)  # (batch*N, d)
            X_embedded = tf.reshape(X_embedded, [batch_size, N, d])  # (batch, N, d)

            # 外部记忆单元 (跨层共享)
            memory_K = tf.get_variable('memory_K', shape=[alpha, d],
                                       initializer=tf.contrib.layers.xavier_initializer())
            memory_V = tf.get_variable('memory_V', shape=[alpha, d],
                                       initializer=tf.contrib.layers.xavier_initializer())

            # 对每个传感器计算外部注意力
            all_outputs = []
            all_attentions = []

            for i in range(N):
                sensor_query = X_embedded[:, i, :]  # (batch, d)

                output, attn = self.external_attention(sensor_query, memory_K, memory_V)
                # output: (batch, d), attn: (batch, alpha)

                all_outputs.append(output)
                all_attentions.append(attn)

            # 堆叠所有传感器的输出
            F_seq_per_sensor = tf.stack(all_outputs, axis=1)  # (batch, N, d)
            F_seq = tf.reduce_mean(F_seq_per_sensor, axis=0)  # (N, d)

            # 计算跨传感器注意力得分矩阵 W_att (N, N)
            attn_stack = tf.stack(all_attentions, axis=1)  # (batch, N, alpha)
            attn_norm = tf.nn.l2_normalize(attn_stack, axis=-1)

            # 计算相似度矩阵
            sim_matrix = tf.matmul(attn_norm, attn_norm, transpose_b=True)  # (batch, N, N)
            W_att = tf.reduce_mean(sim_matrix, axis=0)  # (N, N)

            return F_seq, W_att

    def graph_learning_branch(self, F_seq, W_att, adj_norm):
        """图表示学习分支

        Args:
            F_seq: (N, d) 初始节点特征
            W_att: (N, N) 注意力得分矩阵
            adj_norm: (N, N) 归一化邻接矩阵

        Returns:
            E_node: (N, d) 最终节点表示
        """
        with tf.variable_scope('graph_learning', reuse=tf.AUTO_REUSE):
            N = self.num_sensors
            d = self.ebd_size

            # 可学习的边权重
            W_edge = tf.get_variable('W_edge', shape=[self.num_layers, N, N],
                                     initializer=tf.zeros_initializer())

            # 初始节点嵌入
            node_embeddings = F_seq  # (N, d)
            all_embeddings = [node_embeddings]

            for l in range(self.num_layers):
                # 更新邻接矩阵: A^{(l)} = Norm(A^{(0)} ⊙ σ(W_edge^{(l)}))
                edge_weight = tf.sigmoid(W_edge[l])
                A_l = adj_norm * edge_weight

                # 添加注意力引导: A_agg = Norm(A_l + β * W_att)
                A_agg = A_l + self.beta * W_att

                # 对称归一化
                d_vec = tf.reduce_sum(A_agg, axis=1)
                d_inv_sqrt = tf.pow(d_vec + 1e-8, -0.5)
                d_inv_sqrt = tf.matrix_diag(d_inv_sqrt)
                A_agg_norm = tf.matmul(tf.matmul(d_inv_sqrt, A_agg), d_inv_sqrt)

                # LightGCN消息传递
                node_embeddings = tf.matmul(A_agg_norm, node_embeddings)  # (N, d)
                all_embeddings.append(node_embeddings)

            # 层聚合
            E_node = tf.reduce_mean(tf.stack(all_embeddings, axis=0), axis=0)  # (N, d)

            return E_node

    def dual_branch_fusion(self, F_seq, E_node):
        """双分支输出融合

        Args:
            F_seq: (N, d) 时序分支输出
            E_node: (N, d) 图分支输出

        Returns:
            h_f: (2d) 融合特征向量
            h_seq: (d) 时序全局特征
            h_node: (d) 图全局特征
        """
        # 全局池化
        h_seq = tf.reduce_mean(F_seq, axis=0)  # (d)
        h_node = tf.reduce_mean(E_node, axis=0)  # (d)

        # 拼接
        h_f = tf.concat([h_seq, h_node], axis=0)  # (2d)

        return h_f, h_seq, h_node

    def classifier(self, features):
        """分类器"""
        with tf.variable_scope('classifier', reuse=tf.AUTO_REUSE):
            # Dropout
            features = tf.cond(self.is_training,
                               lambda: tf.nn.dropout(features, 1 - self.dropout),
                               lambda: features)

            # 全连接层
            logits = tf.layers.dense(features, self.num_classes,
                                     activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
        return logits

    def consistency_loss(self, h_seq, h_node):
        """一致性正则化损失"""
        return tf.reduce_mean(tf.square(h_seq - h_node))

    def graph_structure_loss(self):
        """图结构正则化损失 (L1稀疏约束)"""
        with tf.variable_scope('graph_learning', reuse=tf.AUTO_REUSE):
            W_edge = tf.get_variable('W_edge')
            return tf.reduce_sum(tf.abs(W_edge))

    def build_model(self):
        """构建完整模型"""
        # 邻接矩阵归一化
        adj_norm = self.normalize_adjacency(self.adj_init)

        # 时序编码分支
        self.F_seq, self.W_att = self.temporal_encoding_branch(self.X)

        # 图表示学习分支
        self.E_node = self.graph_learning_branch(self.F_seq, self.W_att, adj_norm)

        # 双分支融合
        self.h_f, self.h_seq, self.h_node = self.dual_branch_fusion(self.F_seq, self.E_node)

        # 分类器
        self.logits = self.classifier(self.h_f)
        self.pred_probs = tf.nn.softmax(self.logits)
        self.pred_labels = tf.argmax(self.logits, axis=-1)

    def build_loss(self):
        """构建损失函数"""
        # 分类损失
        self.class_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
        )

        # 一致性损失
        self.cons_loss = self.consistency_loss(self.h_seq, self.h_node)

        # 图结构正则化损失
        self.reg_loss = self.graph_structure_loss()

        # 总损失
        self.total_loss = (self.class_loss +
                           self.lambda_1 * self.cons_loss +
                           self.lambda_2 * self.reg_loss)

    def build_optimizer(self):
        """构建优化器"""
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = self.optimizer.minimize(self.total_loss)

    def build_metrics(self):
        """构建评估指标"""
        correct = tf.equal(self.pred_labels, tf.cast(self.y, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    def normalize_adjacency(self, adj):
        """对称归一化邻接矩阵"""
        adj = adj + tf.eye(tf.shape(adj)[0])
        d = tf.reduce_sum(adj, axis=1)
        d_inv_sqrt = tf.pow(d + 1e-8, -0.5)
        d_inv_sqrt = tf.matrix_diag(d_inv_sqrt)
        adj_norm = tf.matmul(tf.matmul(d_inv_sqrt, adj), d_inv_sqrt)
        return adj_norm

    def train_step(self, sess, batch_X, batch_y, adj_init, lr, dropout):
        """执行一步训练"""
        feed_dict = {
            self.X: batch_X,
            self.y: batch_y,
            self.adj_init: adj_init,
            self.lr: lr,
            self.dropout: dropout,
            self.is_training: True
        }
        loss, acc, _ = sess.run([self.total_loss, self.accuracy, self.train_op], feed_dict)
        return loss, acc

    def eval_step(self, sess, batch_X, batch_y, adj_init):
        """执行一步评估"""
        feed_dict = {
            self.X: batch_X,
            self.y: batch_y,
            self.adj_init: adj_init,
            self.dropout: 0.0,
            self.is_training: False
        }
        loss, acc, preds = sess.run([self.total_loss, self.accuracy, self.pred_labels], feed_dict)
        return loss, acc, preds

    def predict(self, sess, X, adj_init):
        """预测"""
        feed_dict = {
            self.X: X,
            self.adj_init: adj_init,
            self.dropout: 0.0,
            self.is_training: False
        }
        preds = sess.run(self.pred_labels, feed_dict)
        return preds


################## Part 4: 训练与评估 ##################
def train_epoch(model, sess, train_X, train_y, adj_init, config, epoch):
    """训练一个epoch"""
    batches, num_batches = generate_batches(train_X, train_y, config.batch_size, is_train=True)

    # 动态学习率衰减
    lr = config.learning_rate * (0.95 ** epoch)

    total_loss = 0
    total_acc = 0

    for batch_X, batch_y in batches:
        loss, acc = model.train_step(sess, batch_X, batch_y, adj_init, lr, config.dropout_rate)
        total_loss += loss
        total_acc += acc

    return total_loss / num_batches, total_acc / num_batches


def evaluate(model, sess, test_X, test_y, adj_init):
    """评估模型"""
    batches, num_batches = generate_batches(test_X, test_y, 256, is_train=False)

    total_loss = 0
    total_acc = 0
    all_preds = []
    all_labels = []

    for batch_X, batch_y in batches:
        loss, acc, preds = model.eval_step(sess, batch_X, batch_y, adj_init)
        total_loss += loss
        total_acc += acc
        all_preds.extend(preds)
        all_labels.extend(batch_y)

    return total_loss / num_batches, total_acc / num_batches, all_preds, all_labels


def train_few_shot_epoch(model, sess, meta_tasks, adj_init, config, epoch):
    """少样本训练一个epoch"""
    np.random.shuffle(meta_tasks)

    lr = config.learning_rate * (0.95 ** epoch)
    total_loss = 0
    total_acc = 0

    for support_set, query_set in meta_tasks:
        # 准备支撑集
        support_X = np.array([item[0] for item in support_set])
        support_y = np.array([item[1] for item in support_set])

        # 准备查询集
        query_X = np.array([item[0] for item in query_set])
        query_y = np.array([item[1] for item in query_set])

        # 合并训练
        batch_X = np.concatenate([support_X, query_X], axis=0)
        batch_y = np.concatenate([support_y, query_y], axis=0)

        loss, acc = model.train_step(sess, batch_X, batch_y, adj_init, lr, config.dropout_rate)
        total_loss += loss
        total_acc += acc

    return total_loss / len(meta_tasks), total_acc / len(meta_tasks)


################## Part 5: 主程序 ##################
def main():
    print("\n" + "=" * 70)
    print(" " * 20 + "GEA-Net: Lightweight Graph External Attention Network")
    print(" " * 15 + "for Coal Mine Equipment Fault Diagnosis")
    print("=" * 70)

    # 配置参数
    config = ParamConfig()

    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_index

    # 检查数据目录
    data_dir = config.data_dir
    if not os.path.exists(data_dir):
        print(f"\nError: Data directory '{data_dir}' not found!")
        print("Please set the correct data directory in ParamConfig.")
        print("\nExpected directory structure:")
        print("  Data/sensor_data/")
        print("    ├── condition_0/")
        print("    │   ├── fault_0/")
        print("    │   │   ├── sensor_1.csv")
        print("    │   │   ├── sensor_2.csv")
        print("    │   │   └── ...")
        print("    │   └── fault_1/")
        print("    │       └── ...")
        print("    └── condition_1/")
        print("        └── ...")
        return

    # 加载数据
    print("\n[1/5] Loading data...")
    X, y, conditions, sensor_names = load_labeled_dataset(data_dir, config.window_size)

    # 数据归一化
    print("\n[2/5] Preprocessing data...")
    X = normalize_sequence(X)

    # 构建邻接矩阵
    print("\n[3/5] Building adjacency matrix...")
    corr_matrix = compute_correlation_matrix(X)
    adj_init = build_initial_adjacency(sensor_names, corr_matrix, config.threshold_tau)
    adj_init = normalize_adjacency(adj_init)
    print(f"Adjacency matrix shape: {adj_init.shape}")
    print(f"Number of edges: {np.sum(adj_init > 0.1) - len(adj_init)}")

    # 划分训练集和测试集
    print("\n[4/5] Splitting dataset...")
    train_idx, test_idx = train_test_split(
        np.arange(len(X)), test_size=0.2, stratify=y, random_state=seed
    )
    train_X, train_y = X[train_idx], y[train_idx]
    test_X, test_y = X[test_idx], y[test_idx]

    # 进一步划分验证集
    train_idx, val_idx = train_test_split(
        np.arange(len(train_X)), test_size=0.2, stratify=train_y, random_state=seed
    )
    train_X, train_y = train_X[train_idx], train_y[train_idx]
    val_X, val_y = train_X[val_idx], train_y[val_idx]

    num_sensors = X.shape[1]
    num_classes = len(np.unique(y))

    print(f"Training samples: {len(train_X)}")
    print(f"Validation samples: {len(val_X)}")
    print(f"Test samples: {len(test_X)}")
    print(f"Number of sensors: {num_sensors}")
    print(f"Number of fault classes: {num_classes}")

    # 构建模型
    print("\n[5/5] Building GEA-Net model...")
    model = GEA_Net(num_sensors, num_classes, config)

    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)

    # 训练
    with tf.Session(graph=model.graph, config=model.config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        best_val_acc = 0.0

        for epoch in range(config.num_epochs):
            epoch_start = time()

            # 训练
            train_loss, train_acc = train_epoch(model, sess, train_X, train_y, adj_init, config, epoch)

            # 验证
            val_loss, val_acc, _, _ = evaluate(model, sess, val_X, val_y, adj_init)

            epoch_time = time() - epoch_start

            print(f"Epoch {epoch + 1:3d}/{config.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Time: {epoch_time:.2f}s")

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                saver.save(sess, config.check_points, write_meta_graph=False)
                print(f"  -> Best model saved! (Val Acc: {val_acc:.4f})")

        # 加载最佳模型进行测试
        print("\n" + "=" * 70)
        print("Evaluating on test set...")
        print("=" * 70)

        saver.restore(sess, config.check_points)
        test_loss, test_acc, test_preds, test_labels = evaluate(model, sess, test_X, test_y, adj_init)

        print(f"\nTest Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

        # 计算每类准确率
        from sklearn.metrics import classification_report
        print("\nClassification Report:")
        print(classification_report(test_labels, test_preds, digits=4))

    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)


def demo_few_shot():
    """少样本学习演示"""
    print("\n" + "=" * 70)
    print(" " * 15 + "GEA-Net Few-Shot Learning Demo")
    print("=" * 70)

    config = ParamConfig()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_index

    data_dir = config.data_dir
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found. Skipping few-shot demo.")
        return

    # 加载数据
    X, y, conditions, sensor_names = load_labeled_dataset(data_dir, config.window_size)
    X = normalize_sequence(X)

    # 构建邻接矩阵
    corr_matrix = compute_correlation_matrix(X)
    adj_init = build_initial_adjacency(sensor_names, corr_matrix, config.threshold_tau)
    adj_init = normalize_adjacency(adj_init)

    # 按工况划分源域和目标域
    unique_conditions = np.unique(conditions)
    source_conds = unique_conditions[:int(len(unique_conditions) * 0.7)]
    target_conds = unique_conditions[int(len(unique_conditions) * 0.7):]

    source_idx = np.where(np.isin(conditions, source_conds))[0]
    target_idx = np.where(np.isin(conditions, target_conds))[0]

    source_X, source_y = X[source_idx], y[source_idx]
    target_X, target_y = X[target_idx], y[target_idx]

    print(f"Source domain: {len(source_X)} samples")
    print(f"Target domain: {len(target_X)} samples")

    # 创建元任务
    meta_tasks = create_few_shot_tasks(source_X, source_y, conditions[source_idx],
                                       n_ways=5, k_shots=5, n_queries=10, n_tasks=50)
    print(f"Created {len(meta_tasks)} meta-tasks")

    num_sensors = X.shape[1]
    num_classes = len(np.unique(y))

    # 构建模型
    model = GEA_Net(num_sensors, num_classes, config)

    with tf.Session(graph=model.graph, config=model.config) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(50):
            loss, acc = train_few_shot_epoch(model, sess, meta_tasks, adj_init, config, epoch)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}: Loss={loss:.4f}, Acc={acc:.4f}")

    print("Few-shot demo completed!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='GEA-Net for Fault Diagnosis')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'few_shot'], help='Running mode')
    parser.add_argument('--data_dir', type=str, default='./Data/sensor_data',
                        help='Path to dataset directory')

    args = parser.parse_args()

    # 更新数据目录
    config = ParamConfig()
    config.data_dir = args.data_dir

    if args.mode == 'train':
        main()
    elif args.mode == 'few_shot':
        demo_few_shot()