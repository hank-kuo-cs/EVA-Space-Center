import re
import cv2
import json
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
from config import GAMMA_RADIUS, GAMMA_RANGE, SPLIT_DATASET_SIZE, DATASET_SIZE, DATASET_PATH, SUBDIR_NUM


def load_label(label_path, image_name):
    with open(label_path, 'r') as f:
        labels = json.load(f)
    label = labels[image_name]
    label = label[:3] + label[4:6]
    label = np.array(label, dtype=np.double)

    label[0] = (label[0] - GAMMA_RADIUS) / GAMMA_RANGE

    for i in range(1, 5):
        label[i] /= 2 * np.pi

    return label


def load_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.pyrDown(img)

    augmentation = [cv2.equalizeHist, sobel, original]
    n = random.randint(0, 2)

    img = augmentation[n](img)

    return img / 255


def sobel(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    x = cv2.convertScaleAbs(x)
    y = cv2.convertScaleAbs(y)

    return cv2.addWeighted(x, 0.5, y, 0.5, 0)


def original(img):
    return img


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
