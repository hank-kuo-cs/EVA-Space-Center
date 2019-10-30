import time
import argparse
import numpy as np
from torch.utils.data import DataLoader
from glob import glob

from config import *
from net import VGG19
from data import MoonDataset
from loss import get_error_percentage, BCMSELoss
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
    total_error_percentage = 0

    for i in range(LABEL_NUM):
        logging.info('%s error percentage: ' % LABEL_TYPE[i] + str(error_percentage[i]))

        total_error_percentage += error_percentage[i] / LABEL_NUM

    logging.info('total error percentage: ' + str(total_error_percentage))


def set_net_work(model):
    logging.info('Set up network')
    net = VGG19().to(DEVICE)

    if model:
        net.load_state_dict(torch.load(model))
        logging.info('Use this model to test: %s' % str(model))

    return net


def test(loader, dataset_type, model, epoch=-1):
    net = set_net_work(model)

    logging.info('Start testing')
    test_start = time.time()

    error_percentages = np.zeros((1, len(LABEL_TYPE)), dtype=np.double)
    tsne_data, tsne_labels = [], []
    running_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(loader):
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)

            features, outputs = net(images.float())

            add_tsne_data(tsne_data, features[0])
            add_tsne_label(tsne_labels, labels.clone()[0])

            running_loss += BCMSELoss()(outputs.clone().double(), labels.clone()).item()

            for b in range(BATCH_SIZE):
                e_percentage = get_error_percentage(outputs[b].clone(), labels[b].clone())
                error_percentages += e_percentage

            if i % LOG_STEP == LOG_STEP - 1:
                logging.info('%d-th iter, check some predict value:' % (i * BATCH_SIZE))
                logging.info('Predict: ' + str(outputs[0]))
                logging.info('Target: ' + str(labels[0]) + '\n')

    error_percentages /= (DATASET_SIZE[dataset_type] / 100)
    running_loss /= (DATASET_SIZE[dataset_type] // BATCH_SIZE)

    logging.info('Finish testing ' + dataset_type + ' dataset, time = ' + str(time.time() - test_start))
    logging.info('Loss = %.10f' % running_loss)
    print_error_percentage(error_percentages)

    if epoch > 0:
        logging.info('Draw error percentage & tsne onto the tensorboard')

        draw_error_percentage_tensorboard(error_percentages, epoch, dataset_type)
        draw_loss_tensorboard(running_loss, epoch-1, -1, 'test')

        if epoch % LOG_EPOCH == 1:
            draw_tsne_tensorboard(np.array(tsne_data), np.array(tsne_labels), epoch, dataset_type)


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

