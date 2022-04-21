import struct
from array import array
from glob import glob

import numpy as np

from common.img_utils import load_image


class Dataset:
    def __init__(self, dataset_dict):
        self.dataset = dataset_dict

    def add_attr(self, attr_name, attr_data):
        self.dataset[attr_name] = attr_data

    def get_attr(self, attr_name):
        return self.dataset[attr_name]

    def get_subset(self, idxs):
        return Dataset({k: v[idxs] for k, v in self.dataset.items()})


def load_dataset(prefix='/home/yashrahmed/Documents/datasets/kaggle-mnist-as-images/trainingSample/trainingSample'):
    img_file_paths = get_dataset_file_paths(prefix)
    image_labels = list(map(lambda file_path: int(file_path.split('/')[-2]), img_file_paths))
    images = list(map(load_image, img_file_paths))
    dataset = Dataset({
        'label': np.array(image_labels),
        'img_path': np.array(img_file_paths),
        'image': np.array(images)
    })
    return dataset


def load_actual_mnist(folder_path='/home/yashrahmed/Documents/datasets/mnist_actual'):
    test_labels = read_labels_from_ubyte(f'{folder_path}/t10k-labels.idx1-ubyte')
    test_images = read_images_from_ubyte(f'{folder_path}/t10k-images.idx3-ubyte')
    train_labels = read_labels_from_ubyte(f'{folder_path}/train-labels.idx1-ubyte')
    train_images = read_images_from_ubyte(f'{folder_path}/train-images.idx3-ubyte')
    return train_images, train_labels, test_images, test_labels


def get_dataset_file_paths(folder_path):
    return glob(f'{folder_path}/*/*')


def read_labels_from_ubyte(labels_filepath):
    with open(labels_filepath, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = np.array(array("B", file.read()))
        return labels


def read_images_from_ubyte(images_filepath):
    with open(images_filepath, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = np.array(array("B", file.read()))
        return np.reshape(image_data, (size, rows, cols))
