import matplotlib.pyplot as plt
from config import *


def draw_loss_graph(losses, epoch_start=0, step=100):
    y = losses
    x = [i * step + epoch_start * DATASET_SIZE['train'] // BATCH_SiZE for i in range(len(losses))]

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.plot(x, y)
    plt.savefig('./checkpoint/loss_graph/step_%d-%d.png' % (x[0], x[-1]))
