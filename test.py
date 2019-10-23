import os
import sys
import time
import torch
import logging
from net import VGG19
from torch.utils.data import DataLoader
from data import MoonDataset
from config import *
from tqdm import tqdm
from glob import glob


def choose_newest_model():
    model_paths = sorted(glob('./checkpoint/model*'))

    return model_paths[-1]


def print_error_percentage(error_percentages):
    total_error_percentage = 0

    for e in error_percentages:
        total_error_percentage += e

    total_error_percentage /= 3

    print('Gamma error percentage:', error_percentages[0])
    print('Theta error percentage:', error_percentages[1])
    print('Pi error percentage:', error_percentages[2])
    print('Total error percentage:', total_error_percentage)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info('Load data')
    test_set = MoonDataset('test')
    test_loader = DataLoader(test_set, BATCH_SiZE, True, num_workers=2)

    model_path = sys.argv[1] if len(sys.argv) == 2 else choose_newest_model()

    logging.info('Load pretrained model: ' + str(model_path))
    net = VGG19().to(device)
    net.load_state_dict(torch.load(model_path))

    logging.info('Start testing')
    start = time.time()

    error_percentages = [0, 0, 0]

    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels = data[0].to(device), data[1].to(device)

            outputs = net(images.float())

            for i in range(BATCH_SiZE):
                for j in range(3):
                    error_percentages[j] += (outputs[i] - labels[i].float()).item()[j]

    logging.info('Finish testing, time = ' + str(time.time() - start))
    print_error_percentage(error_percentages)
