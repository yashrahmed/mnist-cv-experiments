import cv2
import numpy as np


def show_images(images, disp_name='combined'):
    out_image = np.concatenate(images, axis=1)
    cv2.imshow(disp_name, out_image)
    cv2.waitKey(0)


def load_image(img_path):
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)