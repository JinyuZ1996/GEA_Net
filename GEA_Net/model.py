# model.py

import os
import tensorflow as tf


class GEA_Net:
    def __init__(self, num_sensors, num_classes, config):
        os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_index
        self.graph = tf.Graph()
        self.config_proto = tf.ConfigProto()
        self.config_proto.gpu_options.allow_growth = True

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

    def predict(self, sess, X, adj_init):
        feed_dict = {
            self.X: X,
            self.adj_init: adj_init,
            self.dropout: 0.0,
            self.is_training: False
        }
        preds = sess.run(self.pred_labels, feed_dict)
        return preds