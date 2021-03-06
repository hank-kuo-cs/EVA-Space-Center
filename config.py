import os
import torch
import logging
from net import VGG19, ResNet18, ResNet50

# Basic Setting
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')
IS_PARALLEL = False
if not IS_PARALLEL:
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
PARALLEL_GPUS = [0, 2, 3]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NET_MODEL = VGG19

# PATH
DATASET_PATH = '/data/space/Dataset_all_random_2/'
WRITER_PATH = os.path.expanduser('~') + '/ShowTensorboard/xyz_gamma'

# Dataset
LABEL_TYPE = ['c_x', 'c_y', 'c_z']
LABEL_NUM = len(LABEL_TYPE)
DATASET_TYPE = {'train', 'test', 'validation'}
SPLIT_DATASET_SIZE = {'train': 10000, 'test': 10000, 'validation': 10000}
SUBDIR_NUM = 10
DATASET_SIZE = {'train': 80000, 'test': 10000, 'validation': 10000}

# Loss Function
GAMMA_RADIUS = 1.742887
GAMMA_UNIT = 996.679647
GAMMA_RANGE = 10 / GAMMA_UNIT
CONSTANT_WEIGHT = 20000

# Visualization
LOG_STEP = 100
TSNE_EPOCH = 50
TSNE_STEP = 20
EXPERIMENT_NAME = 'SGD_lr_1e-3'

# hyperparameters
EPOCH_NUM = 300
BATCH_SIZE = 10
LEARNING_RATE = 0.001
MOMENTUM = 0.9
