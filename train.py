import os
import time
import torch
import argparse
import logging
import numpy as np
from glob import glob
from torch.utils.data import DataLoader

from config import NET_MODEL, IS_PARALLEL, DEVICE, PARALLEL_GPUS, LEARNING_RATE, MOMENTUM
from config import EPOCH_NUM, BATCH_SIZE, TSNE_STEP, TSNE_EPOCH, LOG_STEP, DATASET_SIZE
from data import MoonDataset
from loss import MoonLoss
from visualize import draw_loss_tensorboard, draw_tsne_tensorboard, add_tsne_label, add_tsne_data


def set_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='Choose a pretrained model to train')
    parser.add_argument('-s', '--scratch', action='store_true', help='Train model from scratch')

    return parser.parse_args()


def choose_newest_model():
    model_paths = sorted(glob('./checkpoint/model*'))

    return model_paths[-1] if model_paths else None


def get_epoch_num(model):
    try:
        index = model.find('epoch')
        epoch_num = int(model[index + 5: -4])
    except Exception as e:
        logging.info('Cannot find pretrain model, train from scratch')
        epoch_num = 0

    return epoch_num


def set_net_work(model):
    logging.info('Set up network')
    net = NET_MODEL()
    if IS_PARALLEL:
        net = torch.nn.DataParallel(net, device_ids=PARALLEL_GPUS)
    net = net.to(DEVICE)

    if model:
        net.load_state_dict(torch.load(model))
        logging.info('Find pretrained model, continue training this model: ' + str(model))

    return net


def save_net_work(net, epoch):
    if not os.path.exists('checkpoint/'):
        os.makedirs('checkpoint/')

    save_model_path = 'checkpoint/model_epoch%.3d.pth' % (epoch + 1)
    torch.save(net.state_dict(), save_model_path)


def train(data_loader, model):
    net = set_net_work(model)

    criterion = MoonLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    logging.info('Start training')
    train_start = time.time()

    epoch_now = get_epoch_num(model)

    for epoch in range(epoch_now, EPOCH_NUM):
        logging.info('Train epoch %d' % (epoch + 1))
        epoch_start = time.time()

        tsne_data, tsne_labels = [], []
        running_loss, epoch_loss = 0.0, 0.0

        for i, data in enumerate(data_loader):
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

            optimizer.zero_grad()

            features, outputs = net(inputs.float())

            if (i * BATCH_SIZE) % TSNE_STEP == 0:
                add_tsne_data(tsne_data, features[0])
                add_tsne_label(tsne_labels, labels.clone()[0])

            loss = criterion(outputs.double(), labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item()

            if i % LOG_STEP == LOG_STEP - 1:
                running_loss /= LOG_STEP
                logging.info('[%d epoch, %5d step] loss: %.6f' % (epoch + 1, i + 1, running_loss))
                draw_loss_tensorboard(running_loss, epoch, i + 1, 'train')

                running_loss = 0.0

        save_net_work(net, epoch)

        logging.info('Draw loss & tsne onto the tensorboard')
        draw_loss_tensorboard(epoch_loss / (DATASET_SIZE['train'] // BATCH_SIZE), epoch, -1, 'train')

        # if epoch % TSNE_EPOCH == TSNE_EPOCH - 1:
            # draw_tsne_tensorboard(np.array(tsne_data), np.array(tsne_labels), epoch + 1, 'train')

        logging.info('Finish one epoch, time = %s' % str(time.time() - epoch_start))

    logging.info('Finished training, time = %s' % str(time.time() - train_start))


if __name__ == '__main__':
    args = set_argument_parser()

    logging.info('Load data')
    train_set = MoonDataset('train')
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model_path = choose_newest_model() if not args.model else args.model
    model_path = None if args.scratch else model_path

    train(train_loader, model_path)
