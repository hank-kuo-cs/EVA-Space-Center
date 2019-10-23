import os
import time
import torch
import logging
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data import MoonDataset
from net import VGG19
from config import *
from glob import glob


def draw_loss_graph(losses, epoch_start=0, step=100):
    y = losses
    x = [i * step + epoch_start * TRAIN_DATA_SIZE // BATCH_SiZE for i in range(len(losses))]

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.plot(x, y)
    plt.savefig('./checkpoint/loss_graph/step_%d-%d.png' % (x[0], x[-1]))


def choose_newest_model():
    model_paths = sorted(glob('./checkpoint/model*'))

    if not model_paths:
        return None

    return model_paths[-1]


def get_epoch_num(model_path):
    i = model_path.find('epoch')

    return int(model_path[i+5: -4])


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_DEVICE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info('Load data')
    train_set = MoonDataset('train')
    train_loader = DataLoader(train_set, BATCH_SiZE, True, num_workers=2)

    logging.info('Set VGG model')
    net = VGG19().to(device)

    model_path = choose_newest_model()

    if model_path:
        net.load_state_dict(torch.load(model_path))
        logging.info('Find pretrained model, continue training this model')

    epoch_start = get_epoch_num(model_path) if model_path else 0

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    logging.info('Start training')
    train_start = time.time()
    graph_losses = []

    for epoch in range(EPOCH_NUM - epoch_start):
        start = time.time()

        epoch += epoch_start
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs.float())

            loss = criterion(outputs.double(), labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if i % LOG_STEP == LOG_STEP - 1:
                running_loss /= LOG_STEP
                logging.info('[%d epoch, %5d step] loss: %.6f' % (epoch + 1, i + 1, running_loss))

                graph_losses.append(running_loss)
                running_loss = 0.0

        model_path = 'checkpoint/model_epoch%d.pth' % (epoch + 1)
        draw_loss_graph(graph_losses, epoch_start, LOG_STEP)
        torch.save(net.state_dict(), model_path)

        logging.info('Finish one epoch, time = %s' % str(time.time() - start))

    logging.info('Finished training, time = %s' % str(time.time() - train_start))
