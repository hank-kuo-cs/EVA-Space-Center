import os
import torch
import logging
import matplotlib.pyplot as plot
from torch.utils.data import DataLoader
from data import MoonDataset
from net import VGG19
from config import *
from tqdm import tqdm
from glob import glob


def draw_loss_graph(losses, step=1000):
    pass


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

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info('Load data')
    train_set = MoonDataset('train')
    train_loader = DataLoader(train_set, BATCH_SiZE, True, num_workers=2)

    logging.info('Set VGG model')
    net = VGG19().to(device)

    model_path = choose_newest_model()
    if model_path:
        net.load_state_dict(torch.load(model_path))

    epoch_start = get_epoch_num(model_path) if model_path else 0

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    logging.info('Start training')
    graph_losses = []

    for epoch in range(EPOCH_NUM - epoch_start):
        epoch += epoch_start
        running_loss = 0.0

        for i, data in enumerate(tqdm(train_loader)):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs.float())

            loss = criterion(outputs.double(), labels)
            if i % 5 == 0:
                print(loss)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if i % 1000 == 999:
                running_loss /= 1000
                logging.info('[%d epoch, %5d step] loss: %.6f' % (epoch + 1, i + 1, running_loss))

                graph_losses.append(running_loss)
                running_loss = 0.0

        if epoch % 20 == 19 and epoch > 0:
                model_path = 'checkpoint/model_epoch%d.pth' % (epoch + 1)
                torch.save(net.state_dict(), model_path)

    logging.info('Finished training')
