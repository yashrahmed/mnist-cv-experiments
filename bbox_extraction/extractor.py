import cv2
import numpy as np


def get_bbox(image):
    pts_x, pts_y = np.where(image == 255)
    # bbox = cv2.boundingRect(np.vstack((pts_y, pts_x)).transpose()) # Bug in OpenCV version.
    return np.min(pts_x), np.min(pts_y), np.max(pts_x), np.max(pts_y)


def extract_bbox(image, b_box):
    x1, y1, x2, y2 = b_box
    r, c = image.shape
    out_img = np.zeros([r + 4, c + 4], dtype=np.uint8)  # 2px padding
    out_img[2:r + 2, 2:c + 2] = cv2.resize(image[x1:x2, y1:y2], (r, c))
    return out_img


def extract_bbox_region(image):
    b_box = get_bbox(image)
    return extract_bbox(image, b_box)
