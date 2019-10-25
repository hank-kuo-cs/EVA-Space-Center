from tensorboardX import SummaryWriter
from config import *
from data import MoonDataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


writer = SummaryWriter('test_tsne/TSNE')


def draw_tsne_tensorboard(data, labels, epoch, data_type, label_type):
    writer.add_embedding(data, labels, tag='%s/%s/epoch%d/%s' % (EXPERIMENT_NAME, data_type, epoch, label_type))


if __name__ == '__main__':
    test_dataset = MoonDataset('test')
    test_dataloader = DataLoader(test_dataset, BATCH_SIZE, True, num_workers=2)

    imgs = []
    t_labels = []

    for i, data in tqdm(enumerate(test_dataloader)):
        images, labels = data[0], data[1]

        for b in range(BATCH_SIZE):
            img = images[b].view(-1).numpy()
            imgs.append(img)

            label = labels[b][0].item()
            t_labels.append(label)

    imgs = np.array(imgs)

    draw_tsne_tensorboard(imgs, t_labels, 1, 'test', 'gamma')



