import time
import argparse
from glob import glob
from torch.utils.data import DataLoader

from net import VGG19
from config import *
from data import MoonDataset
from loss import BCMSELoss
from visualize import draw_loss_tensorboard, draw_tsne_tensorboard


def set_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Choose a pretrained model to train')
    parser.add_argument('-s', '--scratch', action='store_true', help='Train model from scratch')

    return parser.parse_args()


def choose_newest_model():
    model_paths = sorted(glob('./checkpoint/model*'))

    if not model_paths:
        return None

    return model_paths[-1]


def get_epoch_num(model_path):
    index = model_path.find('epoch')

    return int(model_path[index+5: -4])


def train(train_loader, model_path):
    label_types = ['gamma', 'phi', 'theta']

    logging.info('Set VGG model')
    net = VGG19().to(DEVICE)

    if model_path:
        net.load_state_dict(torch.load(model_path))
        logging.info('Find pretrained model, continue training this model: ' + str(model_path))

    epoch_start = get_epoch_num(model_path) if model_path else 0

    criterion = BCMSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    logging.info('Start training')
    train_start = time.time()

    for epoch in range(EPOCH_NUM - epoch_start):
        tsne_labels = [[], [], []]
        tsne_data = []

        start = time.time()

        epoch += epoch_start

        running_loss, epoch_loss = 0.0, 0.0

        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

            optimizer.zero_grad()

            features, outputs = net(inputs.float())

            tsne_data.append(features[0].detach().cpu().numpy())
            for j in range(3):
                tsne_labels[j].append(labels[0][j].clone().item())

            loss = criterion(outputs.double(), labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item()

            if i % LOG_STEP == LOG_STEP - 1:
                running_loss /= LOG_STEP

                logging.info('[%d epoch, %5d step] loss: %.6f' % (epoch + 1, i + 1, running_loss))
                draw_loss_tensorboard(running_loss, epoch, i, 'train')

                running_loss = 0.0

        model_path = 'checkpoint/model_epoch%.3d.pth' % (epoch + 1)
        torch.save(net.state_dict(), model_path)

        logging.info('Draw loss & tsne onto the tensorboard')
        draw_loss_tensorboard(epoch_loss / (DATASET_SIZE['train'] // BATCH_SIZE), epoch, -1, 'train')
        if epoch % 10 == 9:
            for i in range(3):
                draw_tsne_tensorboard(tsne_data, tsne_labels[i], epoch + 1, 'train', label_types[i])

        logging.info('Finish one epoch, time = %s' % str(time.time() - start))

    logging.info('Finished training, time = %s' % str(time.time() - train_start))


if __name__ == '__main__':
    args = set_argument_parser()

    logging.info('Load data')
    train_set = MoonDataset('train')
    train_loader = DataLoader(train_set, BATCH_SIZE, True, num_workers=2)

    model_path = args.model if args.model else (choose_newest_model() if not args.scratch else None)

    train(train_loader, model_path)
