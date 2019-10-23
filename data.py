import os
import cv2
import torch
import ntpath
import pickle
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
from config import *


DATA_SIZE = {'train': TRAIN_DATA_SIZE, 'test': TEST_DATA_SIZE, 'validation': VALIDATION_SIZE}


def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo)

    return data


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def scale_labels(labels, value):
    return value * labels


def load_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.pyrDown(img)
    img = cv2.equalizeHist(img)

    return img / 255


class MoonDataset(Dataset):
    def __init__(self, data_type):
        self.data_type = data_type
        self.data_size = DATA_SIZE[data_type]
        self.image_files, self.label_files = self.load_data()
        self.data_type = data_type

    def __len__(self):
        return self.data_size

    def __getitem__(self, item):
        file_index = item // SPLIT_DATASET_SIZE[self.data_type]
        file_num = item % SPLIT_DATASET_SIZE[self.data_type]

        image_path = self.image_files[file_index][file_num]
        image = load_image(image_path)

        image_name = path_leaf(image_path)
        label = np.array(unpickle(self.label_files[file_index])[image_name])
        label = scale_labels(label, 1000)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
            ]
        )

        sample = transform(image), torch.from_numpy(label)

        return sample

    def load_data(self):
        data_types = ['train', 'test', 'validation']

        if self.data_type not in data_types:
            raise ValueError('path type must be \'train\', \'test\', or \'validation\', but get \'%s\'' % self.data_type)

        dataset_path = os.path.curdir + '/dataset/%s' % self.data_type

        image_files = []

        for i in range(10):
            imgs_path = dataset_path + '/images/' + str(i) + '/train*'
            image_files.append(glob(imgs_path))

        labels_path = dataset_path + '/labels/gt*'
        label_files = sorted(glob(labels_path))

        return image_files, label_files
