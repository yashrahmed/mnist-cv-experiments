import cv2
import numpy as np


def get_bbox(image):
    pts_x, pts_y = np.where(image == 255)
    # bbox = cv2.boundingRect(np.vstack((pts_y, pts_x)).transpose()) # Bug in OpenCV version.
    return np.min(pts_x), np.min(pts_y), np.max(pts_x), np.max(pts_y)


def align_bbox(image, b_box):
    x1, y1, x2, y2 = b_box
    r, c = image.shape
    out_img = np.zeros([r, c], dtype=np.uint8)  # 2px padding
    out_img[0: x2-x1, 0:y2-y1] = image[x1:x2, y1:y2]
    return out_img


def extract_bbox(image, b_box):
    x1, y1, x2, y2 = b_box
    r, c = image.shape
    out_img = np.zeros([r + 4, c + 4], dtype=np.uint8)  # 2px padding
    out_img[2:r + 2, 2:c + 2] = cv2.resize(image[x1:x2, y1:y2], (r, c))
    return out_img


def extract_bbox_region(image, th=70, mode='extract'):
    th_image = threshold_image(image, th)
    b_box = get_bbox(th_image)
    if mode == 'extract':
        out_img = extract_bbox(image, b_box)
    elif mode == 'align':
        out_img = align_bbox(image, b_box)
    else:
        raise Exception("Unsupported Bounding box operation")
    return out_img


def threshold_image(image, th=70):
    return cv2.threshold(image, th, 255, cv2.THRESH_BINARY)[1]
