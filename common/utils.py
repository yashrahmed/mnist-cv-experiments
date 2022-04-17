from glob import glob

import cv2
import numpy as np


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


def get_dataset_file_paths(folder_path):
    return glob(f'{folder_path}/*/*')


def load_image(img_path):
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)


def show_images(images, disp_name='combined'):
    out_image = np.concatenate(images, axis=1)
    cv2.imshow(disp_name, out_image)
    cv2.waitKey(0)
