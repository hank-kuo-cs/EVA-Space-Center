import re
import cv2
import json
import ntpath
import pickle
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
from config import *


def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo)

    return data


def load_label(label_path, image_name):
    with open(label_path, 'r') as f:
        labels = json.load(f)
    label = np.array(labels[image_name][:6], dtype=np.double)

    label[0] = (label[0] - GAMMA_RADIUS) / GAMMA_RANGE
    label[3] /= GAMMA_RADIUS

    label[1] /= (2 * np.pi)
    label[2] /= (2 * np.pi)
    label[4] /= (2 * np.pi)
    label[5] /= (2 * np.pi)

    return label


def normalize_label(label):
    label[0] = (label[0] - GAMMA_RADIUS) / GAMMA_RANGE
    label[1] /= (2 * np.pi)
    label[2] /= (2 * np.pi)

    return label


def path_leaf(path):
    head, tail = ntpath.split(path)

    return tail or ntpath.basename(head)


def load_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.pyrDown(img)
    img = cv2.equalizeHist(img)

    return img / 255


class MoonDataset(Dataset):
    def __init__(self, data_type):
        self.data_type = data_type
        self.image_files, self.label_files = self.load_data()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        image_path = self.image_files[item]
        image = load_image(image_path)

        image_name = re.split('/', image_path)[-1][:-4]
        target_num = item // SPLIT_DATASET_SIZE[self.data_type]

        label = load_label(self.label_files[target_num], image_name)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
            ]
        )

        sample = transform(image), torch.from_numpy(label)

        return sample

    def load_data(self):
        dataset_path = DATASET_PATH + self.data_type

        image_files = []

        dir_num = DATASET_SIZE[self.data_type] // SPLIT_DATASET_SIZE[self.data_type]

        for i in range(dir_num):
            for j in range(SUBDIR_NUM):
                imgs_path = dataset_path + '/images/' + '%d/%d_%d' % (i, i, j) + '/Dataset*'
                image_files += (sorted(glob(imgs_path)))

        labels_path = dataset_path + '/labels/target*'
        label_files = sorted(glob(labels_path))

        return image_files, label_files
