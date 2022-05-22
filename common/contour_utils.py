import cv2
import numpy as np


def swap_cols(x):
    return np.array(x[:, [1, 0]])


def get_contours(bin_image, method=cv2.CHAIN_APPROX_NONE):
    contours, hierarchy = cv2.findContours(bin_image, cv2.RETR_LIST, method)
    return contours, hierarchy


def get_contours_from_image(bin_image, method=cv2.CHAIN_APPROX_NONE, opencv_fmt=False, dtype=np.uint16):
    contours = get_contours(bin_image, method)[0]
    return sample_points_from_contour(contours, opencv_fmt, dtype)


def sample_points_from_contour(contours, opencv_fmt=False, dtype=np.uint16):
    # Assumes that the input is a simplified contour.
    if opencv_fmt:
        points = np.unique(np.vstack(contours), axis=0)
    else:
        points = swap_cols(np.unique(np.vstack(contours).reshape([-1, 2]), axis=0))
    return points.astype(dtype)
