import json
import cv2
import ntpath
import numpy as np
import shutil
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
from config import *


def check_directory(directory):
    directory_path = os.path.join(DATASET_PATH, directory)
    if not os.path.exists(directory_path):
        logging.info('Create directory {}'.format(directory))
        os.makedirs(directory_path)


def read_json(file):
    file_path = os.path.join(DATASET_PATH, file)
    with open(file_path, 'r') as reader:
        data = json.loads(reader.read())

    # print(data.keys())
    return data


def decompress_targz_file(mode, file):
    if mode == 'train':
        lv1_dir = file
    else:
        lv1_dir = '0'
    for i in range(10):
        file_path = os.path.join(DATASET_PATH, mode, 'images', file + '_{}.tar.gz'.format(i))
        check_directory(os.path.join(DATASET_PATH, mode, 'images', lv1_dir, file + '_{}'.format(i)))
        extract_dir = os.path.join(DATASET_PATH, mode, 'images', lv1_dir, file + '_{}'.format(i))
        shutil.unpack_archive(file_path, extract_dir, 'gztar')
        logging.info('End decompress {}'.format(file_path))


def normalize_label(label):
    label[0] = (label[0] - MOON_RADIUS)
    for i in range(6, 9):
        label[i] += 1
    for i in range(9):
        label[i] /= LIMIT[i]

    return label


def path_leaf(path):
    head, tail = ntpath.split(path)

    return tail or ntpath.basename(head)


def remove_filename_extension(base_name):
    file_name = os.path.splitext(base_name)[0]

    return file_name


def load_image(img_path):
    print(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img.size is None:
        # exit(1)
        img = cv2.cv.LoadImage(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.pyrDown(img)
    img = cv2.equalizeHist(img)

    return img / 255


class MoonDataset(Dataset):
    def __init__(self, data_type):
        self.data_type = data_type
        self.data_size = DATASET_SIZE[data_type]
        self.image_files, self.label_files = self.load_data()

    def __len__(self):
        return self.data_size

    def __getitem__(self, item):
        file_index_lv1 = item // LV_1_SPLIT_DATASET_SIZE[self.data_type]
        file_lv2 = item % LV_1_SPLIT_DATASET_SIZE[self.data_type]
        file_index_lv2 = file_lv2 // LV_2_SPLIT_DATASET_SIZE[self.data_type]
        file_num = file_lv2 % LV_2_SPLIT_DATASET_SIZE[self.data_type]

        image_path = self.image_files[file_index_lv1][file_index_lv2][file_num]
        image = load_image(image_path)

        image_name = remove_filename_extension(path_leaf(image_path))
        label = np.array(read_json(self.label_files[file_index_lv1])[image_name])
        label = normalize_label(label)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
        )

        sample = transform(image), torch.from_numpy(label)

        return sample

    def load_data(self):
        dataset_path = os.path.join(DATASET_PATH, self.data_type)

        image_files = []

        lv1_num = DATASET_SIZE[self.data_type] // LV_1_SPLIT_DATASET_SIZE[self.data_type]
        lv2_num = LV_1_SPLIT_DATASET_SIZE[self.data_type] // LV_2_SPLIT_DATASET_SIZE[self.data_type]

        for i in range(lv1_num):
            lv1_image_file = []
            for j in range(lv2_num):
                imgs_path = os.path.join(dataset_path, 'images', str(i), '{}_{}'.format(i, j), DATASET_NAME + '_*')
                lv1_image_file.append(sorted(glob(imgs_path)))
            image_files.append(lv1_image_file)
            # print(np.array(image_files).shape)

        labels_path = os.path.join(dataset_path, 'labels', 'target_*')
        label_files = sorted(glob(labels_path))

        return image_files, label_files


# if __name__ == '__main__':
    # for i in range(8):
    #     decompress_targz_file('train', str(i))

    # decompress_targz_file('test', '8')
    # decompress_targz_file('valid', '9')
