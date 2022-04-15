from collections import defaultdict
from functools import partial, reduce
from glob import glob

import cv2
import numpy as np


def build_image_lookup(
        prefix='/home/yashrahmed/Documents/datasets/kaggle-mnist-as-images/trainingSample/trainingSample'):
    digit_folder_paths = glob(f'{prefix}/*')
    list_of_digit_img_paths = list(map(get_file_names_in_folder, digit_folder_paths))
    digit_image_paths = reduce(collect_file_names, list_of_digit_img_paths, [])
    img_path_lookup = reduce(group_by_digit, digit_image_paths, defaultdict(list))
    return partial(load_image, img_path_lookup=img_path_lookup)


def collect_file_names(all_names, names):
    all_names.extend(names)
    return all_names


def get_file_names_in_folder(folder_path):
    return glob(f'{folder_path}/*')


def group_by_digit(img_lookup, img_path):
    digit = img_path.split('/')[-2]
    img_lookup[digit].append(img_path)
    return img_lookup


def load_image(img_selector, img_path_lookup):
    (digit, img_idx) = img_selector
    img_path = img_path_lookup[str(digit)][img_idx]
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)


def show_images(images, disp_name='combined'):
    out_image = np.concatenate(images, axis=1)
    cv2.imshow(disp_name, out_image)
    cv2.waitKey(0)


if __name__ == '__main__':
    lookup = build_image_lookup()
    image = lookup((9, 16))
    show_images([image])
