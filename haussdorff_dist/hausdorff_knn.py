import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from bbox_extraction.extractor import extract_bbox_region
from common.contour_utils import get_contours, sample_points_from_contour
from common.dataset_utils import load_actual_mnist

from common.img_utils import show_images, draw_contours_on_image, to_color
from haussdorff_dist.distance import compute_hauss_dist

from numpy.linalg import norm


def compute_cost_matrix(contours_1, contours_2):
    cost_mat = []
    # dist_extractor = cv2.createHausdorffDistanceExtractor()
    for i, c_1 in enumerate(contours_1):
        cost_mat_row = []
        for j, c_2 in enumerate(contours_2):
            d = compute_hauss_dist(c_1, c_2, ratio=0.2)
            # d = norm(aligned_test_images[i] - aligned_train_images[j])
            # d = norm(test_images[i] - train_images[j])
            # d = dist_extractor.computeDistance(c_1, c_2)

            cost_mat_row.append(d)
            # cost_mat_row.append(norm(im_1 - im_2))
        cost_mat.append(cost_mat_row)
    return np.array(cost_mat)


def pre_process(img):
    proc_img = extract_bbox_region(threshold_image(img))
    return sample_points_from_contour(get_contours(proc_img)[0])


def pre_process_non_align(img):
    proc_img = threshold_image(img)
    return sample_points_from_contour(get_contours(proc_img)[0])


def pre_process_tmp(img):
    proc_img = extract_bbox_region(threshold_image(img))
    return proc_img


def run_knn(train_cost_mat, train_label, test_cost_mat, test_label, neighbors=3, weights='distance'):
    knn = KNeighborsClassifier(n_neighbors=neighbors, metric='precomputed', weights=weights, algorithm='auto')
    knn.fit(train_cost_mat, train_label)
    pred_labels = knn.predict(test_cost_mat)
    score = accuracy_score(test_label, pred_labels) * 100
    print(f'N={neighbors},score={score}%')


def threshold_image(image, th=70):
    return cv2.threshold(image, th, 255, cv2.THRESH_BINARY)[1]


if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_actual_mnist()

    n_train = 5000
    n_test = 500
    #
    # # 0 18 == 7 6
    # img = extract_bbox_region(threshold_image(test_images[1]))
    # contours1 = get_contours(img)[0]
    # image = to_color(img)
    # draw_contours_on_image(image, contours1)
    #
    # img2 = extract_bbox_region(threshold_image(train_images[0]))
    # contours2 = get_contours(img2)[0]
    # image2 = to_color(img2)
    # draw_contours_on_image(image2, contours1)
    # draw_contours_on_image(image2, contours2)
    #
    # dist_extractor = cv2.createHausdorffDistanceExtractor()
    #
    # d = dist_extractor.computeDistance(sample_points_from_contour(contours1).reshape([-1, 1, 2]), sample_points_from_contour(contours2).reshape([-1, 1, 2]))
    # d2 = compute_hauss_dist(sample_points_from_contour(contours1), sample_points_from_contour(contours2))
    # print(d)
    # print(d2)
    #
    # show_images([image, image2])

    aligned_train_data = [pre_process(img) for img in train_images[0:n_train]]
    aligned_test_data = [pre_process(img) for img in test_images[0:n_test]]

    test_cost_matrix = compute_cost_matrix(aligned_test_data, aligned_train_data)

    run_knn(np.zeros([3, 3]), train_labels[0:n_train], test_cost_matrix, test_labels[0:n_test])
