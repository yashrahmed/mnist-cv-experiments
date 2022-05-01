import cv2
import numpy as np

from common.dataset_utils import load_actual_mnist
from common.img_utils import show_image, draw_polygons_on_image


def threshold_image(images, threshold=100):
    return np.where(images > threshold, 255, 0).astype(np.uint8)


def get_contours(bin_image):
    contours, hierarchy = cv2.findContours(bin_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy


def get_polygon(contour):
    epsilon = 0.01 * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


def get_polygons(contours):
    return [get_polygon(contour) for contour in contours]


if __name__ == '__main__':
    train_images, train_labels, _, _ = load_actual_mnist()
    image_of_6 = train_images[train_labels == 8][1983]
    image_of_6_col = cv2.cvtColor(image_of_6, cv2.COLOR_GRAY2BGR)
    th_image = threshold_image(image_of_6)
    det_contours, _ = get_contours(th_image)
    # polygons = get_polygons(det_contours)
    # img = draw_polygons_on_image(image_of_6, polygons)
    img = cv2.drawContours(image_of_6_col, det_contours, -1, (0, 0, 255), 1)
    show_image(cv2.resize(img, (300, 300)))
