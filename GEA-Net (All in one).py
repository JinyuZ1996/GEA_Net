# GEA_Net_Complete.py
# Lightweight Graph External Attention Network for Fault Diagnosis
# TensorFlow v1 Implementation

from time import time
import random
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from glob import glob
from sklearn.model_selection import train_test_split

np.seterr(all='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

seed = 2025
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)


class ParamConfig:
    def __init__(self):
        self.learning_rate = 0.01
        self.dropout_rate = 0.1
        self.batch_size = 256
        self.num_epochs = 100
        self.eval_verbose = 10
        self.fast_running = False
        self.fast_ratio = 0.5

        self.embedding_size = 16
        self.num_layers = 2
        self.external_memory_dim = 16
        self.beta = 0.8
        self.lambda_1 = 0.1
        self.lambda_2 = 0.01
        self.window_size = 1024
        self.threshold_tau = 0.6

        self.data_dir = "./Data/sensor_data/"
        self.check_points = "./check_points/GEA_Net.ckpt"
        self.gpu_index = '0'


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


class GEA_Net:
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
        self.X = tf.placeholder(tf.float32,
                                shape=[None, self.num_sensors, self.window_size],
                                name='input_sequences')
        self.y = tf.placeholder(tf.int32, shape=[None], name='labels')
        self.adj_init = tf.placeholder(tf.float32,
                                       shape=[self.num_sensors, self.num_sensors],
                                       name='adjacency_init')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.dropout = tf.placeholder(tf.float32, name='dropout_rate')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

    def external_attention(self, queries, memory_K, memory_V):
        scores = tf.matmul(queries, memory_K, transpose_b=True)
        scores = scores / tf.sqrt(tf.cast(tf.shape(queries)[-1], tf.float32))
        attention = tf.nn.l2_normalize(scores, axis=-1)
        output = tf.matmul(attention, memory_V)
        return output, attention

    def temporal_encoding_branch(self, X):
        with tf.variable_scope('temporal_encoding', reuse=tf.AUTO_REUSE):
            batch_size = tf.shape(X)[0]
            N = self.num_sensors
            d = self.ebd_size
            alpha = self.external_mem_dim

            X_reshaped = tf.reshape(X, [-1, self.window_size])
            W_proj = tf.get_variable('W_proj', shape=[self.window_size, d],
                                     initializer=tf.contrib.layers.xavier_initializer())
            X_embedded = tf.matmul(X_reshaped, W_proj)
            X_embedded = tf.reshape(X_embedded, [batch_size, N, d])

            memory_K = tf.get_variable('memory_K', shape=[alpha, d],
                                       initializer=tf.contrib.layers.xavier_initializer())
            memory_V = tf.get_variable('memory_V', shape=[alpha, d],
                                       initializer=tf.contrib.layers.xavier_initializer())

            all_outputs = []
            all_attentions = []

            for i in range(N):
                sensor_query = X_embedded[:, i, :]
                output, attn = self.external_attention(sensor_query, memory_K, memory_V)
                all_outputs.append(output)
                all_attentions.append(attn)

            F_seq_per_sensor = tf.stack(all_outputs, axis=1)
            F_seq = tf.reduce_mean(F_seq_per_sensor, axis=0)

            attn_stack = tf.stack(all_attentions, axis=1)
            attn_norm = tf.nn.l2_normalize(attn_stack, axis=-1)
            sim_matrix = tf.matmul(attn_norm, attn_norm, transpose_b=True)
            W_att = tf.reduce_mean(sim_matrix, axis=0)

            return F_seq, W_att

    def graph_learning_branch(self, F_seq, W_att, adj_norm):
        with tf.variable_scope('graph_learning', reuse=tf.AUTO_REUSE):
            N = self.num_sensors
            d = self.ebd_size

            W_edge = tf.get_variable('W_edge', shape=[self.num_layers, N, N],
                                     initializer=tf.zeros_initializer())

            node_embeddings = F_seq
            all_embeddings = [node_embeddings]

            for l in range(self.num_layers):
                edge_weight = tf.sigmoid(W_edge[l])
                A_l = adj_norm * edge_weight
                A_agg = A_l + self.beta * W_att

                d_vec = tf.reduce_sum(A_agg, axis=1)
                d_inv_sqrt = tf.pow(d_vec + 1e-8, -0.5)
                d_inv_sqrt = tf.matrix_diag(d_inv_sqrt)
                A_agg_norm = tf.matmul(tf.matmul(d_inv_sqrt, A_agg), d_inv_sqrt)

                node_embeddings = tf.matmul(A_agg_norm, node_embeddings)
                all_embeddings.append(node_embeddings)

            E_node = tf.reduce_mean(tf.stack(all_embeddings, axis=0), axis=0)

            return E_node

    def dual_branch_fusion(self, F_seq, E_node):
        h_seq = tf.reduce_mean(F_seq, axis=0)
        h_node = tf.reduce_mean(E_node, axis=0)
        h_f = tf.concat([h_seq, h_node], axis=0)
        return h_f, h_seq, h_node

    def classifier(self, features):
        with tf.variable_scope('classifier', reuse=tf.AUTO_REUSE):
            features = tf.cond(self.is_training,
                               lambda: tf.nn.dropout(features, 1 - self.dropout),
                               lambda: features)
            logits = tf.layers.dense(features, self.num_classes,
                                     activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
        return logits

    def consistency_loss(self, h_seq, h_node):
        return tf.reduce_mean(tf.square(h_seq - h_node))

    def graph_structure_loss(self):
        with tf.variable_scope('graph_learning', reuse=tf.AUTO_REUSE):
            W_edge = tf.get_variable('W_edge')
            return tf.reduce_sum(tf.abs(W_edge))

    def build_model(self):
        adj_norm = self.normalize_adjacency(self.adj_init)
        self.F_seq, self.W_att = self.temporal_encoding_branch(self.X)
        self.E_node = self.graph_learning_branch(self.F_seq, self.W_att, adj_norm)
        self.h_f, self.h_seq, self.h_node = self.dual_branch_fusion(self.F_seq, self.E_node)
        self.logits = self.classifier(self.h_f)
        self.pred_probs = tf.nn.softmax(self.logits)
        self.pred_labels = tf.argmax(self.logits, axis=-1)

    def build_loss(self):
        self.class_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
        )
        self.cons_loss = self.consistency_loss(self.h_seq, self.h_node)
        self.reg_loss = self.graph_structure_loss()
        self.total_loss = (self.class_loss +
                           self.lambda_1 * self.cons_loss +
                           self.lambda_2 * self.reg_loss)

    def build_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = self.optimizer.minimize(self.total_loss)

    def build_metrics(self):
        correct = tf.equal(self.pred_labels, tf.cast(self.y, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    def normalize_adjacency(self, adj):
        adj = adj + tf.eye(tf.shape(adj)[0])
        d = tf.reduce_sum(adj, axis=1)
        d_inv_sqrt = tf.pow(d + 1e-8, -0.5)
        d_inv_sqrt = tf.matrix_diag(d_inv_sqrt)
        adj_norm = tf.matmul(tf.matmul(d_inv_sqrt, adj), d_inv_sqrt)
        return adj_norm

    def train_step(self, sess, batch_X, batch_y, adj_init, lr, dropout):
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
        feed_dict = {
            self.X: batch_X,
            self.y: batch_y,
            self.adj_init: adj_init,
            self.dropout: 0.0,
            self.is_training: False
        }
        loss, acc, preds = sess.run([self.total_loss, self.accuracy, self.pred_labels], feed_dict)
        return loss, acc, preds


def train_epoch(model, sess, train_X, train_y, adj_init, config, epoch):
    batches, num_batches = generate_batches(train_X, train_y, config.batch_size, is_train=True)
    lr = config.learning_rate * (0.95 ** epoch)
    total_loss = 0
    total_acc = 0
    for batch_X, batch_y in batches:
        loss, acc = model.train_step(sess, batch_X, batch_y, adj_init, lr, config.dropout_rate)
        total_loss += loss
        total_acc += acc
    return total_loss / num_batches, total_acc / num_batches


def evaluate(model, sess, test_X, test_y, adj_init):
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


def main():
    print("\n" + "=" * 70)
    print(" " * 20 + "GEA-Net: Lightweight Graph External Attention Network")
    print(" " * 15 + "for Coal Mine Equipment Fault Diagnosis")
    print("=" * 70)

    config = ParamConfig()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_index

    data_dir = config.data_dir
    if not os.path.exists(data_dir):
        print(f"\nError: Data directory '{data_dir}' not found!")
        print("Please set the correct data directory in ParamConfig.")
        return

    print("\n[1/5] Loading data...")
    X, y, conditions, sensor_names = load_labeled_dataset(data_dir, config.window_size)

    print("\n[2/5] Preprocessing data...")
    X = normalize_sequence(X)

    print("\n[3/5] Building adjacency matrix...")
    corr_matrix = compute_correlation_matrix(X)
    adj_init = build_initial_adjacency(sensor_names, corr_matrix, config.threshold_tau)
    adj_init = normalize_adjacency(adj_init)
    print(f"Adjacency matrix shape: {adj_init.shape}")
    print(f"Number of edges: {np.sum(adj_init > 0.1) - len(adj_init)}")

    print("\n[4/5] Splitting dataset...")
    train_idx, test_idx = train_test_split(
        np.arange(len(X)), test_size=0.2, stratify=y, random_state=seed
    )
    train_X, train_y = X[train_idx], y[train_idx]
    test_X, test_y = X[test_idx], y[test_idx]

    train_idx, val_idx = train_test_split(
        np.arange(len(train_X)), test_size=0.2, stratify=train_y, random_state=seed
    )
    val_X, val_y = train_X[val_idx], train_y[val_idx]
    train_X, train_y = train_X[train_idx], train_y[train_idx]

    num_sensors = X.shape[1]
    num_classes = len(np.unique(y))

    print(f"Training samples: {len(train_X)}")
    print(f"Validation samples: {len(val_X)}")
    print(f"Test samples: {len(test_X)}")
    print(f"Number of sensors: {num_sensors}")
    print(f"Number of fault classes: {num_classes}")

    print("\n[5/5] Building GEA-Net model...")
    model = GEA_Net(num_sensors, num_classes, config)

    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)

    with tf.Session(graph=model.graph, config=model.config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        best_val_acc = 0.0

        for epoch in range(config.num_epochs):
            epoch_start = time()

            train_loss, train_acc = train_epoch(model, sess, train_X, train_y, adj_init, config, epoch)
            val_loss, val_acc, _, _ = evaluate(model, sess, val_X, val_y, adj_init)

            epoch_time = time() - epoch_start

            print(f"Epoch {epoch + 1:3d}/{config.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Time: {epoch_time:.2f}s")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                saver.save(sess, config.check_points, write_meta_graph=False)
                print(f"  -> Best model saved! (Val Acc: {val_acc:.4f})")

        print("\n" + "=" * 70)
        print("Evaluating on test set...")
        print("=" * 70)

        saver.restore(sess, config.check_points)
        test_loss, test_acc, test_preds, test_labels = evaluate(model, sess, test_X, test_y, adj_init)

        print(f"\nTest Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

        from sklearn.metrics import classification_report
        print("\nClassification Report:")
        print(classification_report(test_labels, test_preds, digits=4))

    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='GEA-Net for Fault Diagnosis')
    parser.add_argument('--data_dir', type=str, default='./Data/sensor_data',
                        help='Path to dataset directory')
    args = parser.parse_args()

    config = ParamConfig()
    config.data_dir = args.data_dir

    main()