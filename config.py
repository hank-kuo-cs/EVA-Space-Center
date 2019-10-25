import os
import torch
import logging

# Basic Setting
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# PATH
DATASET_PATH = os.path.expanduser('~') + '/dataset_random/'
WRITER_PATH = os.path.expanduser('~') + '/runs/Space-Center'
EXPERIMENT_NAME = 'BCMSE_SGD_lr=1e-3'

# Dataset
SPLIT_DATASET_SIZE = {'train': 10000, 'test': 10000, 'validation': 10000}
DATASET_SIZE = {'train': 80000, 'test': 10000, 'validation': 10000}

# Loss Function
GAMMA_RADIUS = 1.742887
GAMMA_UNIT = 996.679647
GAMMA_RANGE = 14 / GAMMA_UNIT
GAMMA_WEIGHT = 447.30877957

# Visualization
LOG_STEP = 100

# hyperparameters
EPOCH_NUM = 500
BATCH_SIZE = 10
LEARNING_RATE = 0.001
MOMENTUM = 0.9
