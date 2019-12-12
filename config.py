import os
import torch
import numpy as np
import logging
from net import VGG19, ResNet18, ResNet50

# Basic Setting
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
IS_PARALLEL = False
PARALLEL_GPUS = [0, 2, 3]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NET_MODEL = ResNet18

# Dataset
DATASET_NAME = 'Dataset_all_random_2'
LABEL_TYPE = ['c_gamma', 'c_theta', 'c_phi', 'p_gamma', 'p_theta', 'p_phi', 'u_x', 'u_y', 'u_z']
LABEL_NUM = len(LABEL_TYPE)
DATASET_TYPE = {'train', 'test', 'validation'}
LV_1_SPLIT_DATASET_SIZE = {'train': 1000, 'test': 10000, 'validation': 10000}
LV_2_SPLIT_DATASET_SIZE = {'train': 1000, 'test': 1000, 'validation': 1000}
DATASET_SIZE = {'train': 1000, 'test': 10000, 'validation': 10000}

# PATH
CONSTANT_PENALITY_MODE = 'dcp'  # 'dcp', 'cp', 'little_cp', 'no_cp'
SAVE_POINT = 'find_defect_images'
DATASET_PATH = '/data/space/' + DATASET_NAME
WRITER_PATH = os.path.expanduser('~') + '/Tensorboard/' + SAVE_POINT

# Units
UNIT_REAL = 996.679647  # 1 OpenGL unit = 996.679647 km
MOON_RADIUS = 1.742887  # in OpenGL unit
OPENGL_1_METER = 0.001 / UNIT_REAL
pi = np.pi

# Constraints
VIEWPORT = [800, 600]
FOVY = 90.0  # in degrees
Z_NEAR = 1.0
Z_FAR = 100.0
LOWER_BOUND = MOON_RADIUS + (OPENGL_1_METER * 200)  # 200m above moon surface
UPPER_BOUND = MOON_RADIUS + (OPENGL_1_METER * 10000)   # 10,000m above moon surface
LIMIT = [OPENGL_1_METER * 10000, 2 * pi, 2 * pi, MOON_RADIUS, 2 * pi, 2 * pi, 2.0, 2.0, 2.0]
CONSTANT_WEIGHT = 20000

# Visualization
LOG_STEP = 100
TSNE_EPOCH = 50
TSNE_STEP = 20
EXPERIMENT_NAME = 'CosSimiBCLoss_Adam_lr_1e-3'

# hyperparameters
EPOCH_NUM = 300
BATCH_SIZE = 30
LEARNING_RATE = 0.001
MOMENTUM = 0.9
