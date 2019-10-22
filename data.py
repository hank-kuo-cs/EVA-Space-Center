import os
import cv2
import torch
import ntpath
import pickle
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms


def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')

    return data


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def scale_labels(labels, value):
    return value * labels


def load_data(path_type):
    images, labels = [], []
    path_types = ['train', 'test', 'validation']

    if path_type not in path_types:
        raise ValueError('path type must be \'train\', \'test\', or \'validation\', but get \'%s\'' % path_type)

    dataset_path = os.path.curdir + '/dataset/%s' % path_type

    label_paths = sorted(glob(dataset_path + '/labels/gt*'))

    for i in range(10):
        img_paths = glob(dataset_path + '/images/' + str(i) + '/train*')

        for img_path in img_paths:
            images.append(cv2.imread(img_path))

            image_name = path_leaf(img_path)
            label = (unpickle(label_paths[i]))[bytes(image_name, encoding="utf8")]
            labels.append(label)

    return np.array(images), np.array(labels)


def load_images(path):
    imgs = []

    for i in range(10):
        img_paths = glob(path + str(i) + '/*')

        for img_path in img_paths:
            imgs.append(cv2.imread(img_path))
            print(os.curdir)

    return imgs


def load_labels(path):
    labels = []

    label_paths = glob(path)

    for label_path in label_paths:
        label = unpickle(label_path)
        labels.append(label)

    return labels


class MoonDataset(Dataset):
    def __init__(self, data_type):
        self.imgs, self.labels = load_data(data_type)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        sample = (self.imgs[item], torch.from_numpy(self.labels)[item])

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]
        )

        sample = transform(sample[0]), sample[1]

        return sample
