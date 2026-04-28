# config.py

import os


class ParamConfig:
    def __init__(self):
        # block 1: 训练超参数
        self.learning_rate = 0.01
        self.dropout_rate = 0.1
        self.batch_size = 256
        self.num_epochs = 100
        self.eval_verbose = 10
        self.fast_running = False
        self.fast_ratio = 0.5

        # block 2: 模型超参数
        self.embedding_size = 16
        self.num_layers = 2
        self.external_memory_dim = 16
        self.beta = 0.8
        self.lambda_1 = 0.1
        self.lambda_2 = 0.01
        self.window_size = 1024
        self.threshold_tau = 0.6

        # block 3: 路径配置
        self.data_dir = "./Data/sensor_data/"
        self.check_points = "./check_points/GEA_Net.ckpt"
        self.gpu_index = '0'

        # block 4: 随机种子
        self.seed = 2025


config = ParamConfig()