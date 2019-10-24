import time
import argparse
import numpy as np
from net import VGG19
from torch.utils.data import DataLoader
from data import MoonDataset
from config import *
from tqdm import tqdm
from glob import glob
from visualize import draw_error_percentage_tensorboard


def set_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Choose a model to test')
    parser.add_argument('-am', '--all_model', action='store_true',
                        help='Use all models to test, and recording on the tensorboard')

    return parser.parse_args()


def get_epoch_num(model_path):
    index = model_path.find('epoch')

    return int(model_path[index+5: -4])


def get_newest_model():
    model_paths = sorted(glob('./checkpoint/model*'))

    return model_paths[-1]


def get_all_model():
    return sorted(glob('./checkpoint/model*'))


def print_error_percentage(error_percentage):
    total_error_percentage = 0

    for e in error_percentage:
        total_error_percentage += e

    total_error_percentage /= 3

    print('Gamma error percentage:', error_percentage[0])
    print('Theta error percentage:', error_percentage[1])
    print('Pi error percentage:', error_percentage[2])
    print('Total error percentage:', total_error_percentage)


def test(model_path, epoch=-1):
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

    error_percentages = np.array(error_percentages) / DATASET_SIZE['test'] * 100

    logging.info('Finish testing, time = ' + str(time.time() - start))

    print_error_percentage(error_percentages)

    if epoch > 0:
        draw_error_percentage_tensorboard(error_percentages, epoch)


if __name__ == '__main__':
    args = set_argument_parser()

    logging.info('Load data')
    test_set = MoonDataset('test')
    test_loader = DataLoader(test_set, BATCH_SiZE, True, num_workers=2)

    model_path = args.model if args.model else get_newest_model()

    if not args.all_model:
        test(model_path)
    else:
        for model in get_newest_model():
            test(model, get_epoch_num(model))

