from glob import glob

import cv2
import numpy as np
import pandas as pd


def load_dataset(prefix='/home/yashrahmed/Documents/datasets/kaggle-mnist-as-images/trainingSample/trainingSample'):
    img_file_paths = get_dataset_file_paths(prefix)
    image_labels = list(map(lambda file_path: int(file_path.split('/')[-2]), img_file_paths))
    df = pd.DataFrame({
        'label': image_labels,
        'img_path': img_file_paths
    })
    return df


def get_dataset_file_paths(folder_path):
    return glob(f'{folder_path}/*/*')


def load_image(img_path):
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)


def show_images(images, disp_name='combined'):
    out_image = np.concatenate(images, axis=1)
    cv2.imshow(disp_name, out_image)
    cv2.waitKey(0)
