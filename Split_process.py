import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from time import time
from collections import defaultdict
import logging
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 设置随机种子
np.random.seed(2025)
tf.set_random_seed(2025)

# 导入自定义模块
from GEA_Net import ParamConfig, GraphExternalAttention, GEANetTrainer
from data_utils import (
    load_labeled_dataset,
    build_adjacency_from_positions,
    compute_correlation_matrix,
    create_few_shot_tasks
)


class DataPreprocessor:
    """数据预处理器"""

    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()

    def normalize_sequence(self, X):
        """对每个传感器的时间序列进行Z-score归一化

        Args:
            X: (num_samples, num_sensors, window_size)

        Returns:
            X_norm: 归一化后的数据
        """
        X_norm = X.copy()
        num_samples, num_sensors, window_size = X.shape

        for i in range(num_samples):
            for j in range(num_sensors):
                mean = np.mean(X[i, j, :])
                std = np.std(X[i, j, :])
                if std > 1e-6:
                    X_norm[i, j, :] = (X[i, j, :] - mean) / std

        return X_norm

    def add_gaussian_noise(self, X, noise_level=0.01):
        """添加高斯噪声进行数据增强"""
        noise = np.random.randn(*X.shape) * noise_level
        return X + noise

    def time_shift_augmentation(self, X, max_shift=10):
        """时间位移数据增强"""
        shift = np.random.randint(-max_shift, max_shift)
        if shift > 0:
            X_shifted = np.roll(X, shift, axis=-1)
            X_shifted[:, :, :shift] = 0
        elif shift < 0:
            X_shifted = np.roll(X, shift, axis=-1)
            X_shifted[:, :, shift:] = 0
        else:
            X_shifted = X
        return X_shifted

    def apply_augmentation(self, X, y, augmentation_ratio=0.5):
        """应用数据增强"""
        num_samples = len(X)
        aug_indices = np.random.choice(num_samples,
                                       size=int(num_samples * augmentation_ratio),
                                       replace=False)

        X_aug = []
        y_aug = []

        for idx in aug_indices:
            # 随机选择增强方式
            aug_type = np.random.choice(['noise', 'shift', 'both'])

            X_sample = X[idx]

            if aug_type in ['noise', 'both']:
                X_sample = self.add_gaussian_noise(X_sample)
            if aug_type in ['shift', 'both']:
                X_sample = self.time_shift_augmentation(X_sample)

            X_aug.append(X_sample)
            y_aug.append(y[idx])

        X_aug = np.array(X_aug)
        y_aug = np.array(y_aug)

        # 合并原始数据和增强数据
        X_combined = np.concatenate([X, X_aug], axis=0)
        y_combined = np.concatenate([y, y_aug], axis=0)

        return X_combined, y_combined


class ConditionSplitter:
    """工况划分器"""

    def __init__(self, config):
        self.config = config

    def split_by_condition(self, X, y, conditions, train_ratio=0.7):
        """按工况划分源域和目标域

        Args:
            X: 数据
            y: 标签
            conditions: 工况标识
            train_ratio: 训练工况比例

        Returns:
            source_X, source_y, source_cond: 源域数据
            target_X, target_y, target_cond: 目标域数据
        """
        unique_conditions = np.unique(conditions)
        num_train_conds = max(1, int(len(unique_conditions) * train_ratio))

        # 随机选择训练工况
        train_conds = np.random.choice(unique_conditions, num_train_conds, replace=False)
        test_conds = np.setdiff1d(unique_conditions, train_conds)

        source_indices = np.where(np.isin(conditions, train_conds))[0]
        target_indices = np.where(np.isin(conditions, test_conds))[0]

        source_X = X[source_indices]
        source_y = y[source_indices]
        source_cond = conditions[source_indices]

        target_X = X[target_indices]
        target_y = y[target_indices]
        target_cond = conditions[target_indices]

        print(f"Source domain: {len(source_X)} samples, {len(train_conds)} conditions")
        print(f"Target domain: {len(target_X)} samples, {len(test_conds)} conditions")

        return source_X, source_y, source_cond, target_X, target_y, target_cond

    def create_cross_validation_splits(self, X, y, conditions, n_folds=5):
        """创建交叉验证划分"""
        splits = []
        unique_conditions = np.unique(conditions)

        for fold in range(n_folds):
            # 轮换作为目标域的工况
            test_conds = [unique_conditions[fold % len(unique_conditions)]]
            train_conds = np.setdiff1d(unique_conditions, test_conds)

            train_indices = np.where(np.isin(conditions, train_conds))[0]
            test_indices = np.where(np.isin(conditions, test_conds))[0]

            splits.append({
                'train': train_indices,
                'test': test_indices,
                'train_conds': train_conds,
                'test_conds': test_conds
            })

        return splits


class FaultDiagnosisPipeline:
    """故障诊断完整流程"""

    def __init__(self, config):
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        self.splitter = ConditionSplitter(config)
        self.model = None
        self.trainer = None
        self.adj_init = None
        self.sensor_names = None

    def load_and_preprocess_data(self, data_path):
        """加载并预处理数据"""
        print("=" * 60)
        print("Step 1: Loading and preprocessing data")
        print("=" * 60)

        # 加载数据
        X, y, conditions, self.sensor_names = load_labeled_dataset(
            data_path,
            window_size=self.config.window_size,
            step_size=self.config.window_size // 2
        )

        print(f"Original dataset shape: {X.shape}")
        print(f"Number of fault types: {len(np.unique(y))}")
        print(f"Number of conditions: {len(np.unique(conditions))}")

        # 数据归一化
        X = self.preprocessor.normalize_sequence(X)

        # 数据增强
        X, y = self.preprocessor.apply_augmentation(X, y, augmentation_ratio=0.3)
        print(f"After augmentation: {X.shape}")

        # 计算相关系数矩阵
        corr_matrix = compute_correlation_matrix(X)

        # 构建邻接矩阵（使用相关性和物理位置）
        # 注：如果没有物理位置信息，可以只使用相关性
        self.adj_init = build_adjacency_from_positions(
            self.sensor_names,
            sensor_positions=None,  # 如果没有物理位置信息，设为None
            correlation_matrix=corr_matrix,
            threshold=self.config.threshold_tau
        )

        print(f"Adjacency matrix shape: {self.adj_init.shape}")
        print(f"Number of edges: {np.sum(self.adj_init > 0) - len(self.adj_init)}")

        return X, y, conditions

    def train_standard_mode(self, X, y, conditions):
        """标准训练模式"""
        print("\n" + "=" * 60)
        print("Step 2: Training in standard mode")
        print("=" * 60)

        # 划分训练集和验证集（随机划分，不区分工况）
        train_indices, val_indices = train_test_split(
            np.arange(len(X)), test_size=0.2, stratify=y, random_state=2025
        )

        train_X = X[train_indices]
        train_y = y[train_indices]
        val_X = X[val_indices]
        val_y = y[val_indices]

        print(f"Training set: {len(train_X)} samples")
        print(f"Validation set: {len(val_X)} samples")

        num_sensors = X.shape[1]
        num_classes = len(np.unique(y))

        # 构建模型
        self.model = GraphExternalAttention(self.config, num_sensors, num_classes)
        self.trainer = GEANetTrainer(self.config, self.model, None)

        # 训练
        self.trainer.train_standard(
            train_X, train_y, val_X, val_y, self.adj_init
        )

    def train_few_shot_mode(self, X, y, conditions, k_shots=[1, 3, 5]):
        """少样本训练模式"""
        print("\n" + "=" * 60)
        print("Step 2: Training in few-shot mode")
        print("=" * 60)

        # 按工况划分源域和目标域
        source_X, source_y, source_cond, target_X, target_y, target_cond = \
            self.splitter.split_by_condition(X, y, conditions, train_ratio=0.7)

        print(f"Source domain: {len(source_X)} samples")
        print(f"Target domain: {len(target_X)} samples")

        num_sensors = X.shape[1]
        num_classes = len(np.unique(y))

        results = {}

        for k_shot in k_shots:
            print(f"\n--- Testing {k_shot}-shot scenario ---")

            # 从源域创建元任务
            meta_tasks = create_few_shot_tasks(
                source_X, source_y, source_cond,
                n_ways=min(5, num_classes),
                k_shots=k_shot,
                n_queries=10,
                n_tasks=100
            )

            print(f"Created {len(meta_tasks)} meta-tasks")

            # 从目标域创建支撑集和测试集
            support_X, test_X, support_y, test_y = self.create_few_shot_support(
                target_X, target_y, target_cond, k_shot
            )

            # 构建模型
            self.model = GraphExternalAttention(self.config, num_sensors, num_classes)
            self.trainer = GEANetTrainer(self.config, self.model, None)

            # 元训练
            self.trainer.train_few_shot(meta_tasks, self.adj_init)

            # 在目标域上评估
            # 注意：需要重新加载模型或使用原型网络进行适配
            # 这里简化处理，实际应使用原型网络进行少样本分类

            results[f'{k_shot}-shot'] = {
                'support_size': len(support_X),
                'test_size': len(test_X)
            }

        return results

    def create_few_shot_support(self, target_X, target_y, target_cond, k_shot):
        """从目标域创建少样本支撑集"""
        unique_labels = np.unique(target_y)
        support_X = []
        support_y = []
        test_X = []
        test_y = []

        for label in unique_labels:
            label_indices = np.where(target_y == label)[0]
            np.random.shuffle(label_indices)

            # 选择k_shot个样本作为支撑集
            support_indices = label_indices[:min(k_shot, len(label_indices))]
            # 其余作为测试集
            test_indices = label_indices[min(k_shot, len(label_indices)):]

            for idx in support_indices:
                support_X.append(target_X[idx])
                support_y.append(label)

            for idx in test_indices:
                test_X.append(target_X[idx])
                test_y.append(label)

        support_X = np.array(support_X)
        support_y = np.array(support_y)
        test_X = np.array(test_X)
        test_y = np.array(test_y)

        print(f"Support set: {len(support_X)} samples ({len(unique_labels)} classes × {k_shot})")
        print(f"Test set: {len(test_X)} samples")

        return support_X, test_X, support_y, test_y

    def run_cross_validation(self, X, y, conditions, n_folds=5):
        """运行交叉验证"""
        print("\n" + "=" * 60)
        print("Step: Running cross-validation")
        print("=" * 60)

        splits = self.splitter.create_cross_validation_splits(X, y, conditions, n_folds)

        fold_results = []
        num_sensors = X.shape[1]
        num_classes = len(np.unique(y))

        for fold, split in enumerate(splits):
            print(f"\n--- Fold {fold + 1}/{n_folds} ---")
            print(f"Train conditions: {split['train_conds']}")
            print(f"Test conditions: {split['test_conds']}")

            train_X = X[split['train']]
            train_y = y[split['train']]
            test_X = X[split['test']]
            test_y = y[split['test']]

            # 划分验证集
            train_X, val_X, train_y, val_y = train_test_split(
                train_X, train_y, test_size=0.2, stratify=train_y, random_state=2025
            )

            # 构建模型
            model = GraphExternalAttention(self.config, num_sensors, num_classes)
            trainer = GEANetTrainer(self.config, model, None)

            # 训练
            trainer.train_standard(train_X, train_y, val_X, val_y, self.adj_init)

            # 评估
            test_acc, _, _ = trainer.evaluate(test_X, test_y, self.adj_init)
            fold_results.append(test_acc)

            print(f"Fold {fold + 1} test accuracy: {test_acc:.4f}")

            # 清理图
            tf.reset_default_graph()

        # 统计结果
        mean_acc = np.mean(fold_results)
        std_acc = np.std(fold_results)

        print(f"\nCross-validation results:")
        print(f"Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

        return fold_results

    def predict_on_new_data(self, new_X, model_checkpoint=None):
        """对新数据进行预测"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")

        if model_checkpoint and os.path.exists(model_checkpoint):
            # 加载保存的模型
            saver = tf.train.Saver()
            with tf.Session() as sess:
                saver.restore(sess, model_checkpoint)
                # 进行预测
                preds = self.trainer.predict(new_X, self.adj_init)
                return preds
        else:
            # 使用当前模型进行预测
            preds = self.trainer.predict(new_X, self.adj_init)
            return preds


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='GEA-Net for Fault Diagnosis')

    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--mode', type=str, default='standard',
                        choices=['standard', 'few_shot', 'cross_val'],
                        help='Training mode')
    parser.add_argument('--k_shots', type=str, default='1,3,5',
                        help='Comma-separated k-shot values for few-shot learning')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--embedding_size', type=int, default=16,
                        help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of GCN layers')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device index')

    return parser.parse_args()


def main():
    """主程序"""
    print("\n" + "=" * 70)
    print(" " * 20 + "GEA-Net: Lightweight Graph External Attention Network")
    print(" " * 15 + "for Coal Mine Equipment Fault Diagnosis")
    print("=" * 70)

    # 解析参数
    args = parse_arguments()

    # 配置参数
    config = ParamConfig()
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.embedding_size = args.embedding_size
    config.num_layers = args.num_layers
    config.learning_rate = args.learning_rate
    config.gpu_index = args.gpu

    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_index

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('gea_net_training.log'),
            logging.StreamHandler()
        ]
    )

    # 创建输出目录
    os.makedirs(os.path.dirname(config.check_points), exist_ok=True)

    # 初始化pipeline
    pipeline = FaultDiagnosisPipeline(config)

    try:
        # 加载和预处理数据
        X, y, conditions = pipeline.load_and_preprocess_data(args.data_dir)

        # 根据模式进行训练
        if args.mode == 'standard':
            pipeline.train_standard_mode(X, y, conditions)

        elif args.mode == 'few_shot':
            k_shots = [int(k) for k in args.k_shots.split(',')]
            results = pipeline.train_few_shot_mode(X, y, conditions, k_shots)
            print(f"\nFew-shot results: {results}")

        elif args.mode == 'cross_val':
            results = pipeline.run_cross_validation(X, y, conditions, n_folds=5)
            print(f"\nCross-validation results: {results}")

        print("\n" + "=" * 70)
        print(" " * 25 + "Training completed successfully!")
        print("=" * 70)

    except Exception as e:
        logging.error(f"Error during execution: {str(e)}")
        raise

    # 清理
    tf.reset_default_graph()


def create_demo_data(data_dir, num_samples=1000):
    """创建演示用数据（用于测试）"""
    import os

    # 创建目录结构
    conditions = ['condition_0', 'condition_1']
    faults = ['fault_0', 'fault_1', 'fault_2', 'fault_3', 'fault_4', 'fault_5']

    for cond in conditions:
        for fault in faults:
            fault_dir = os.path.join(data_dir, cond, fault)
            os.makedirs(fault_dir, exist_ok=True)

            # 生成模拟传感器数据
            for sensor_id in range(6):  # 6个传感器
                file_path = os.path.join(fault_dir, f'sensor_{sensor_id + 1}.csv')

                # 生成模拟时间序列
                t = np.arange(0, num_samples) / 12800  # 采样率12.8kHz

                # 不同故障类型生成不同的信号模式
                fault_id = int(fault.split('_')[-1])
                cond_id = int(cond.split('_')[-1])

                # 模拟信号
                signal = np.sin(2 * np.pi * 100 * t)  # 基频100Hz
                signal += 0.5 * np.sin(2 * np.pi * 200 * t)  # 谐波

                # 添加故障特征
                if fault_id > 0:
                    # 故障频率
                    fault_freq = 50 + fault_id * 30 + cond_id * 10
                    signal += 0.8 * np.sin(2 * np.pi * fault_freq * t)

                # 添加高斯噪声
                signal += 0.1 * np.random.randn(num_samples)

                # 保存CSV（使用分号分隔）
                df = pd.DataFrame({
                    'Time(s)': t,
                    'CH 1': signal,
                    'CH 2': signal * 0.8 + 0.1 * np.random.randn(num_samples),
                    'CH 3': signal * 1.2 - 0.05 * np.random.randn(num_samples)
                })
                df.to_csv(file_path, sep=';', index=False, float_format='%.6f')

            print(f"Created {fault_dir}")

    print(f"\nDemo data created at: {data_dir}")


if __name__ == '__main__':
    # 检测是否有命令行参数
    if len(sys.argv) > 1:
        main()
    else:
        # 如果没有参数，运行演示模式
        print("No arguments provided. Running in demo mode...")
        print("To run with actual data, use: python main.py --data_dir <path_to_data> --mode standard")

        # 创建临时演示数据
        demo_data_dir = "./demo_data"
        if not os.path.exists(demo_data_dir):
            create_demo_data(demo_data_dir, num_samples=2000)

        # 使用演示数据运行
        sys.argv = [
            'main.py',
            '--data_dir', demo_data_dir,
            '--mode', 'standard',
            '--batch_size', '32',
            '--num_epochs', '10'
        ]
        main()