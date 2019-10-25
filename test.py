import time
import argparse
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from glob import glob

from config import *
from net import VGG19
from data import MoonDataset
from loss import get_error_percentage, BCMSELoss
from visualize import draw_error_percentage_tensorboard


def set_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Choose a model to test')
    parser.add_argument('-v', '--validation', action='store_true', help='Test the validation set')
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
    error_type = ['gamma', 'phi', 'theta']
    total_error_percentage = 0

    for i in range(3):
        print('%s error percentage:' % error_type[i], error_percentage[i])

        total_error_percentage += error_percentage[i] / 3

    print('total error percentage:', total_error_percentage)


def test(test_type, model_path, epoch=-1):
    net = VGG19().to(DEVICE)
    net.load_state_dict(torch.load(model_path))

    logging.info('Start testing')
    start = time.time()

    error_percentages = np.array([0.0, 0.0, 0.0])
    avg_loss = 0.0

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)

            outputs = net(images.float())
            avg_loss += BCMSELoss()(outputs.clone().double(), labels.clone()).item()

            for b in range(BATCH_SIZE):
                e_percentage = get_error_percentage(outputs[b].clone(), labels[b].clone())
                error_percentages += e_percentage

            if i % LOG_STEP == LOG_STEP - 1:
                logging.info('\n\nCheck some predict value:')
                logging.info('Predict: ' + str(outputs[0]))
                logging.info('Target: ' + str(labels[0]))

    error_percentages /= (DATASET_SIZE[test_type] / 100)
    avg_loss /= (DATASET_SIZE[test_type] // BATCH_SIZE)

    logging.info('Finish testing ' + test_type + ' dataset, time = ' + str(time.time() - start))
    logging.info('Average loss = ' + str(avg_loss))

    print_error_percentage(error_percentages)

    if epoch > 0:
        draw_error_percentage_tensorboard(error_percentages, epoch, test_type)


if __name__ == '__main__':
    args = set_argument_parser()

    test_type = 'test' if not args.validation else 'validation'

    logging.info('Load data')
    test_set = MoonDataset(test_type)
    test_loader = DataLoader(test_set, BATCH_SIZE, True, num_workers=2)

    model_path = args.model if args.model else get_newest_model()
    logging.info('Load model ' + str(model_path))

    if not args.all_model:
        test(test_type, model_path)
    else:
        for model in get_newest_model():
            test(test_type, model, get_epoch_num(model))

