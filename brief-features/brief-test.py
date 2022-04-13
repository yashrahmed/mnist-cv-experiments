from collections import defaultdict
from functools import partial
from glob import glob

import cv2


def build_image_lookup(
        prefix='/home/yashrahmed/Documents/datasets/kaggle-mnist-as-images/trainingSample/trainingSample'):
    img_path_lookup = defaultdict(list)
    for digit_folder_path in glob(f'{prefix}/*'):
        digit = digit_folder_path.split('/')[-1]
        for digit_image_path in glob(f'{digit_folder_path}/*'):
            img_path_lookup[digit].append(digit_image_path)
    return partial(load_image, img_path_lookup=img_path_lookup)


def load_image(digit, img_num, img_path_lookup):
    img_path = img_path_lookup[str(digit)][img_num]
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)


if __name__ == '__main__':
    lookup = build_image_lookup()
    cv2.imshow('img', lookup(3, 23))
    cv2.waitKey(0)
