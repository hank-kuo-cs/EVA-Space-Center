import time
import torch
import logging
import argparse
import numpy as np
from torch.utils.data import DataLoader
from glob import glob

from config import DATASET_SIZE, TSNE_STEP, TSNE_EPOCH, BATCH_SIZE, \
    LABEL_TYPE, LABEL_NUM, NET_MODEL, DEVICE, LOG_STEP, IS_PARALLEL, PARALLEL_GPUS
from data import MoonDataset
from loss import get_error_percentage, MoonLoss, get_gamma
from visualize import draw_error_percentage_tensorboard, draw_tsne_tensorboard, draw_loss_tensorboard, add_tsne_label, add_tsne_data


def set_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Choose a model to test')
    parser.add_argument('-v', '--validation', action='store_true', help='Test the validation set')
    parser.add_argument('-am', '--all_model', action='store_true',
                        help='Use all models to test, and recording on the tensorboard')

    return parser.parse_args()


def get_epoch_num(model):
    index = model.find('epoch')

    return int(model[index+5: -4])


def get_newest_model():
    model_paths = sorted(glob('./checkpoint/model*'))

    return model_paths[-1]


def get_all_model():
    return sorted(glob('./checkpoint/model*'))


def print_error_percentage(error_percentage):
    gamma_km_error = 0

    logging.info('%s error percentage (80km): ' % LABEL_TYPE[0] + str(error_percentage[0]))
    logging.info('%s error percentage (10km): ' % LABEL_TYPE[0] + str(error_percentage[0] * 80 / 10))

    gamma_km_error = error_percentage[0] * 80

    logging.info('gamma error km: ' + str(gamma_km_error))


def set_net_work(model):
    logging.info('Set up network')
    net = NET_MODEL().to(DEVICE)
    if IS_PARALLEL:
        net = torch.nn.DataParallel(net, device_ids=PARALLEL_GPUS)

    if model:
        net.load_state_dict(torch.load(model))
        logging.info('Use this model to test: %s' % str(model))

    return net


def test(loader, dataset_type, model, epoch=-1):
    net = set_net_work(model)

    logging.info('Start testing')
    test_start = time.time()

    error_percentages = np.zeros(LABEL_NUM, dtype=np.double)
    error_percentages_10km = np.zeros(LABEL_NUM, dtype=np.double)
    tsne_data, tsne_labels = [], []
    running_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(loader):
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)

            features, outputs = net(images.float())

            if (i * BATCH_SIZE) % TSNE_STEP == 0:
                add_tsne_data(tsne_data, features[0])
                add_tsne_label(tsne_labels, labels.clone()[0])

            running_loss += MoonLoss()(outputs.clone().double(), labels.clone()).item()

            for b in range(BATCH_SIZE):
                e_percentage = get_error_percentage(outputs[b].clone(), labels[b].clone())
                error_percentages += e_percentage

                error_percentages_10km += e_percentage * 80 / 10

            if i % LOG_STEP == LOG_STEP - 1:
                logging.info('%d-th iter, check some predict value:' % (i * BATCH_SIZE))
                logging.info('Predict: ' + str(outputs[0]) + str(get_gamma(outputs[0])))
                logging.info('Target: ' + str(labels[0]) + '\n')

    error_percentages /= (DATASET_SIZE[dataset_type] / 100)
    error_percentages_10km /= (DATASET_SIZE[dataset_type] / 100)
    running_loss /= (DATASET_SIZE[dataset_type] // BATCH_SIZE)

    logging.info('Finish testing ' + dataset_type + ' dataset, time = ' + str(time.time() - test_start))
    logging.info('Loss = %.10f' % running_loss)
    print_error_percentage(error_percentages)

    if epoch > 0:
        logging.info('Draw error percentage & tsne onto the tensorboard')

        draw_error_percentage_tensorboard(error_percentages_10km, epoch, dataset_type)
        draw_loss_tensorboard(running_loss, epoch-1, -1, 'test')

        # if epoch % TSNE_EPOCH == 1:
            # draw_tsne_tensorboard(np.array(tsne_data), np.array(tsne_labels), epoch, dataset_type)


if __name__ == '__main__':
    args = set_argument_parser()

    dataset_type = 'test' if not args.validation else 'validation'

    logging.info('Load data')
    test_set = MoonDataset(dataset_type)
    test_loader = DataLoader(test_set, BATCH_SIZE, True, num_workers=2)

    model_path = args.model if args.model else get_newest_model()

    if not args.all_model:
        test(test_loader, dataset_type, model_path)
    else:
        for model in get_all_model():
            test(test_loader, dataset_type, model, get_epoch_num(model))
