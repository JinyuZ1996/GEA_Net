# main.py

import os
import numpy as np
from sklearn.model_selection import train_test_split

from config import config
from data_utils import (
    load_labeled_dataset, normalize_sequence, compute_correlation_matrix,
    build_initial_adjacency, normalize_adjacency, set_seed
)
from model import GEA_Net
from trainer import Trainer


def main(data_dir=None):
    if data_dir is not None:
        config.data_dir = data_dir

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_index
    set_seed(config.seed)

    print("\n" + "=" * 70)
    print(" " * 20 + "GEA-Net: Lightweight Graph External Attention Network")
    print(" " * 15 + "for Coal Mine Equipment Fault Diagnosis")
    print("=" * 70)

    if not os.path.exists(config.data_dir):
        print(f"\nError: Data directory '{config.data_dir}' not found!")
        return None, None

    print("\n[1/5] Loading data...")
    X, y, conditions, sensor_names = load_labeled_dataset(
        config.data_dir, config.window_size
    )

    print("\n[2/5] Preprocessing data...")
    X = normalize_sequence(X)

    print("\n[3/5] Building adjacency matrix...")
    corr_matrix = compute_correlation_matrix(X)
    adj_init = build_initial_adjacency(
        sensor_names, corr_matrix, config.threshold_tau
    )
    adj_init = normalize_adjacency(adj_init)
    print(f"Adjacency matrix shape: {adj_init.shape}")
    print(f"Number of edges: {np.sum(adj_init > 0.1) - len(adj_init)}")

    print("\n[4/5] Splitting dataset...")
    train_idx, test_idx = train_test_split(
        np.arange(len(X)), test_size=0.2, stratify=y, random_state=config.seed
    )
    train_X, train_y = X[train_idx], y[train_idx]
    test_X, test_y = X[test_idx], y[test_idx]

    train_idx, val_idx = train_test_split(
        np.arange(len(train_X)), test_size=0.2, stratify=train_y, random_state=config.seed
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
    trainer = Trainer(model, config)

    best_val_acc, test_acc = trainer.run(
        train_X, train_y, val_X, val_y, test_X, test_y, adj_init
    )

    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)

    return best_val_acc, test_acc


if __name__ == '__main__':
    main()