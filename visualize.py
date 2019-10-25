import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from config import *


writer = SummaryWriter(WRITER_PATH)


def draw_loss_graph(losses, epoch_start=0, step=100):
    y = losses
    x = [i * step + epoch_start * DATASET_SIZE['train'] // BATCH_SIZE for i in range(len(losses))]

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.plot(x, y)
    plt.savefig('./checkpoint/loss_graph/step_%d-%d.png' % (x[0], x[-1]))


def draw_loss_tensorboard(loss, epoch, step):
    x = epoch + 1 if step < 0 else step + epoch * (DATASET_SIZE['train'] // BATCH_SIZE)
    y = loss
    tag = '%s/train/loss_with_%s' % (EXPERIMENT_NAME, 'epoch' if step < 0 else 'step')

    writer.add_scalar(tag, y, x)


def draw_error_percentage_tensorboard(error_percentage, epoch, test_type):
    error_type = ['gamma', 'phi', 'theta']

    total_error_percentage = 0

    for e in error_percentage:
        total_error_percentage += e

    total_error_percentage /= 3

    for i in range(3):
        tag = '%s/%s/%s_error_percentage' % (EXPERIMENT_NAME, test_type, error_type[i])

        writer.add_scalar(tag, error_percentage[i], epoch)

    tag = '%s/%s/total_error_percentage' % (EXPERIMENT_NAME, test_type)

    writer.add_scalar(tag, total_error_percentage, epoch)
