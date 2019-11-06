from tensorboardX import SummaryWriter
from config import *


writer = SummaryWriter(WRITER_PATH)


def draw_loss_tensorboard(loss, epoch, step, dataset_type):
    x = epoch + 1 if step < 0 else step + epoch * (DATASET_SIZE[dataset_type] // BATCH_SIZE)
    y = loss
    tag = '%s/%s/loss_with_%s' % (EXPERIMENT_NAME, dataset_type, 'epoch' if step < 0 else 'step')

    writer.add_scalar(tag, y, x)


def draw_error_percentage_tensorboard(error_percentage, epoch, dataset_type):
    total_error_percentage = 0

    for e in error_percentage:
        total_error_percentage += e

    total_error_percentage /= LABEL_NUM

    for i in range(LABEL_NUM):
        tag = '%s/%s/%s_error_percentage' % (EXPERIMENT_NAME, dataset_type, LABEL_TYPE[i])

        writer.add_scalar(tag, error_percentage[i], epoch)

    tag = '%s/%s/total_error_percentage' % (EXPERIMENT_NAME, dataset_type)

    writer.add_scalar(tag, total_error_percentage, epoch)


def draw_tsne_tensorboard(data, labels, epoch, dataset_type):
    for i in range(labels.shape[1]):
        writer.add_embedding(data, labels[:, i], tag='%s/%s/epoch%d/%s' % (EXPERIMENT_NAME, dataset_type, epoch, LABEL_TYPE[i]))


def add_tsne_data(tsne_data, feature):
    if DEVICE == 'cpu':
        data = feature.detach().numpy() if feature.requires_grad else feature.numpy()
    else:
        data = feature.detach().cpu().numpy() if feature.requires_grad else feature.cpu().numpy()

    tsne_data.append(data)

    return tsne_data


def add_tsne_label(tsne_labels, label):
    labels = []
    for i in range(LABEL_NUM):
        labels.append(label[i].item())

    tsne_labels.append(labels)

    return tsne_labels
