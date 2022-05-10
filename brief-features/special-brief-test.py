import cv2
import numpy as np

from common.dataset_utils import load_actual_mnist
from common.img_utils import draw_brief_features_on_image, show_images


def compute_custom_brief(image, location_pairs):
    # for now loc_pairs is just a list of points to calculate the image sum difference
    # loc_pairs is a list of pairs of the form [[x1, y1, x'1, y`1],.....,[xn, yn, x'n, y`n]]
    # should smooth sums be used??
    sq_radius = 4
    feature = []
    for loc_pair in location_pairs:
        x1, y1, x2, y2 = loc_pair
        p1_ext = [x1 - sq_radius, x1 + sq_radius + 1, y1 - sq_radius, y1 + sq_radius + 1]
        p2_ext = [x2 - sq_radius, x2 + sq_radius + 1, y2 - sq_radius, y2 + sq_radius + 1]
        p1_sum = np.sum(image[x1 - sq_radius:x1 + sq_radius + 1, y1 - sq_radius:y1 + sq_radius + 1])
        p2_sum = np.sum(image[x2 - sq_radius:x2 + sq_radius + 1, y2 - sq_radius:y2 + sq_radius + 1])
        print(f'p1_sum={p1_sum} p2_sum={p2_sum} at loc_pair={loc_pair} with p1_ext={p1_ext} p2_ext={p2_ext}')
        feature.append(p1_sum > p2_sum)
    return np.array(feature, dtype=np.byte)


def compute_custom_brief_intg(image, location_pairs):
    # for now loc_pairs is just a list of points to calculate the image sum difference
    # loc_pairs is a list of pairs of the form [[x1, y1, x'1, y`1],.....,[xn, yn, x'n, y`n]]
    # should smooth sums be used??
    sq_radius = 4
    intg_img = cv2.integral(image)
    print(image.shape)
    print(intg_img.shape)
    feature = []
    for loc_pair in location_pairs:
        x1, y1, x2, y2 = loc_pair
        # coords offset by +1 as the integral image has an extra row and column or 0's at the left
        # and the top of the image.
        p1_ext = [x1 - sq_radius, x1 + sq_radius, y1 - sq_radius, y1 + sq_radius]
        p2_ext = [x2 - sq_radius, x2 + sq_radius, y2 - sq_radius, y2 + sq_radius]
        p1_sum = intg_img[x1 + sq_radius + 1][y1 + sq_radius + 1] \
                 + intg_img[x1 - sq_radius][y1 - sq_radius] \
                 - intg_img[x1 + sq_radius + 1][y1 - sq_radius] \
                 - intg_img[x1 - sq_radius][y1 + sq_radius + 1]
        p2_sum = intg_img[x2 + sq_radius + 1][y2 + sq_radius + 1] \
                 + intg_img[x2 - sq_radius][y2 - sq_radius] \
                 - intg_img[x2 + sq_radius + 1][y2 - sq_radius] \
                 - intg_img[x2 - sq_radius][y2 + sq_radius + 1]
        print(f'p1_sum={p1_sum} p2_sum={p2_sum} at loc_pair={loc_pair} with p1_ext={p1_ext} p2_ext={p2_ext}')
        feature.append(p1_sum > p2_sum)
    return np.array(feature, dtype=np.byte)


if __name__ == '__main__':
    loc_pairs = [[5, 5, 20, 20], [15, 20, 7, 7], [23, 5, 5, 23]]
    train_image, train_label, test_image, test_label = load_actual_mnist()
    brief_features = compute_custom_brief(train_image[0], loc_pairs)
    print(f'brief_features={brief_features}')
    print("""
    ========================================================
    """)
    brief_features_intg = compute_custom_brief_intg(train_image[0], loc_pairs)
    print(f'brief_features_intg={brief_features_intg}')
    out_image = cv2.resize(draw_brief_features_on_image(train_image[0], loc_pairs, resize_value=(28, 28)), (200, 200))
    show_images([out_image])
