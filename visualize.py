import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from config import *


writer = SummaryWriter(WRITER_PATH)


def draw_loss_graph(losses, epoch_start=0, step=100):
    y = losses
    x = [i * step + epoch_start * DATASET_SIZE['train'] // BATCH_SiZE for i in range(len(losses))]

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.plot(x, y)
    plt.savefig('./checkpoint/loss_graph/step_%d-%d.png' % (x[0], x[-1]))


def draw_loss_tensorboard(loss, data_type, epoch, step):
    x = epoch + 1 if step < 0 else step + epoch * (DATASET_SIZE[data_type] // BATCH_SiZE)
    y = loss
    tag = '%s/%s_loss_with_%s' % (EXPERIMENT_NAME, data_type, 'epoch' if step < 0 else 'step')

    writer.add_scalar(tag, y, x)
