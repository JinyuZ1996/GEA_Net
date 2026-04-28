# run.py

import argparse
from main import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GEA-Net for Fault Diagnosis')
    parser.add_argument('--data_dir', type=str, default='./Data/sensor_data',
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--embedding_size', type=int, default=16,
                        help='Embedding dimension')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device index')

    args = parser.parse_args()

    from config import config

    config.data_dir = args.data_dir
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.embedding_size = args.embedding_size
    config.learning_rate = args.learning_rate
    config.gpu_index = args.gpu

    main()