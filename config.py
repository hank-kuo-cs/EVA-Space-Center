import os
import torch
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASET_PATH = '~/dataset_random/'
SPLIT_DATASET_SIZE = {'train': 10000, 'test': 10000, 'valid': 10000}
DATASET_SIZE = {'train': 80000, 'test': 10000, 'validation': 10000}
LOG_STEP = 100

# hyperparameters
EPOCH_NUM = 500
BATCH_SiZE = 10
LEARNING_RATE = 0.001
MOMENTUM = 0.9
