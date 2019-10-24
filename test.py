import sys
import time
import numpy as np
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
    logging.info('Load data')
    test_set = MoonDataset('test')
    test_loader = DataLoader(test_set, BATCH_SiZE, True, num_workers=2)

    model_path = sys.argv[1] if len(sys.argv) == 2 else choose_newest_model()

    logging.info('Load pretrained model: ' + str(model_path))
    net = VGG19().to(DEVICE)
    net.load_state_dict(torch.load(model_path))

    logging.info('Start testing')
    start = time.time()

    error_percentages = [0, 0, 0]

    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)

            outputs = net(images.float())

            for i in range(BATCH_SiZE):
                for j in range(3):
                    error_percentages[j] += (outputs[i] - labels[i].float())[j].item() / labels[i][j].item()

    logging.info('Finish testing, time = ' + str(time.time() - start))
    print_error_percentage(np.array(error_percentages) / DATASET_SIZE['test'])
