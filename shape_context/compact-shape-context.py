import cv2
import numpy as np

from bbox_extraction.bbox_extractor import get_bbox_of_points
from common.contour_utils import get_contours_from_image
from common.dataset_utils import load_actual_mnist
from common.img_utils import threshold_image, draw_rects_on_image, to_color, show_image


def build_compact_sc(image):
    contours = get_contours_from_image(threshold_image(image, 90))
    x_min, y_min, x_max, y_max = get_bbox_of_points(contours)
    print(x_min, y_min, x_max, y_max)
    out_img = to_color(image)
    draw_rects_on_image(out_img, [[x_min, y_min, x_max, y_max]])
    show_image(out_img)


def main():
    train_images, train_labels, test_images, test_labels = load_actual_mnist()
    build_compact_sc(train_images[35])


if __name__ == '__main__':
    main()
