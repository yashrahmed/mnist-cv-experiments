import cv2
import numpy as np


def swap_cols(x):
    return np.array(x[:, [1, 0]])


def get_contours(bin_image):
    contours, hierarchy = cv2.findContours(bin_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return contours, hierarchy


def sample_points_from_contour(contours):
    # Assumes that the input is a simplified contour.
    return swap_cols(np.unique(np.vstack(contours).astype(np.uint16).reshape([-1, 2]), axis=0))
