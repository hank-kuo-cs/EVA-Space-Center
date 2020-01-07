import os
import torch
import logging
from net import VGG19, ResNet18, ResNet50

# Basic Setting
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')
IS_PARALLEL = True
if not IS_PARALLEL:
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
PARALLEL_GPUS = [0, 2, 3]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NET_MODEL = VGG19

# PATH
DATASET_PATH = '/data/space/Dataset_six_random/'
WRITER_PATH = os.path.expanduser('~') + '/ShowTensorboard/gamma'

# Dataset
LABEL_TYPE = ['c_gamma']
LABEL_NUM = len(LABEL_TYPE)
DATASET_TYPE = {'train', 'test', 'validation'}
SPLIT_DATASET_SIZE = {'train': 20000, 'test': 20000, 'validation': 20000}
SUBDIR_NUM = 10
DATASET_SIZE = {'train': 160000, 'test': 20000, 'validation': 20000}

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
BATCH_SIZE = 20
LEARNING_RATE = 0.001
MOMENTUM = 0.9
