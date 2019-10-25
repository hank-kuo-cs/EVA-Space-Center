import os
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# path
DATASET_PATH = os.path.expanduser('~') + '/dataset_random/'
WRITER_PATH = os.path.expanduser('~') + '/runs/Space-Center'
EXPERIMENT_NAME = 'many_data_low_epoch'

# dataset
SPLIT_DATASET_SIZE = {'train': 10000, 'test': 10000, 'validation': 10000}
DATASET_SIZE = {'train': 80000, 'test': 10000, 'validation': 10000}
SCALAR_LABEL = 1
GAMMA_RADIUS = 1.742887
GAMMA_UNIT = 996.679647
GAMMA_RANGE = 14 / GAMMA_UNIT


# visualization
LOG_STEP = 100

# hyperparameters
EPOCH_NUM = 500
BATCH_SIZE = 10
LEARNING_RATE = 0.001
MOMENTUM = 0.9
