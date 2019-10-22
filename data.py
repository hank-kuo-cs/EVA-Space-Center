import os
import cv2
import torch
import ntpath
import pickle
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
from config import DATA_SIZE


def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo)

    return data


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def scale_labels(labels, value):
    return value * labels


def load_data(path_type):
    path_types = ['train', 'test', 'validation']

    if path_type not in path_types:
        raise ValueError('path type must be \'train\', \'test\', or \'validation\', but get \'%s\'' % path_type)

    dataset_path = os.path.curdir + '/dataset/%s' % path_type

    image_files = []

    for i in range(8):
        imgs_path = dataset_path + '/images/' + str(i) + '/train*'
        image_files.append(glob(imgs_path))

    labels_path = dataset_path + '/labels/gt*'
    label_files = sorted(glob(labels_path))

    return image_files, label_files


def load_image(img_path):
    img = cv2.imread(img_path)
    img = np.array(img / 255)
    img.astype(dtype=float)

    return img


def load_label(img_path):
    label_paths = sorted(glob(os.path.curdir + '/dataset/%s' % path_type + '/labels/gt*'))

    image_name = path_leaf(img_path)
    label = (unpickle(label_paths[i]))[image_name]

    return label


class MoonDataset(Dataset):
    def __init__(self, data_type):
        self.image_files, self.label_files = load_data(data_type)
        self.data_type = data_type

    def __len__(self):
        return DATA_SIZE

    def __getitem__(self, item):
        file_index = item // 1
        file_num = item % 1

        image_path = self.image_files[file_index][file_num]
        image = load_image(image_path)

        image_name = path_leaf(image_path)
        label = np.array(unpickle(self.label_files[file_index])[image_name])
        label = scale_labels(label, 1000)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]
        )

        sample = transform(image), torch.from_numpy(label)

        return sample
