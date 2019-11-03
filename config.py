import os
import torch
import logging
from net import VGG19, ResNet101, ResNet152

# Basic Setting
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
IS_PARALLEL = False
PARALLEL_GPUS = [0, 2]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NET_MODEL = ResNet152

# PATH
DATASET_PATH = os.path.expanduser('~') + '/dataset_random/'
WRITER_PATH = os.path.expanduser('~') + '/Tensorboard/Resnet'

# Dataset
LABEL_TYPE = ['gamma', 'phi', 'theta']
LABEL_NUM = len(LABEL_TYPE)
DATASET_TYPE = {'train', 'test', 'validation'}
SPLIT_DATASET_SIZE = {'train': 10000, 'test': 10000, 'validation': 10000}
DATASET_SIZE = {'train': 80000, 'test': 10000, 'validation': 10000}

# Loss Function
GAMMA_RADIUS = 1.742887
GAMMA_UNIT = 996.679647
GAMMA_RANGE = 15 / GAMMA_UNIT
CONSTANT_WEIGHT = 20000

# Visualization
LOG_STEP = 100
LOG_EPOCH = 50
EXPERIMENT_NAME = 'BCMSE_SGD_lr_1e-3'

# hyperparameters
EPOCH_NUM = 300
BATCH_SIZE = 10
LEARNING_RATE = 0.001
MOMENTUM = 0.9
