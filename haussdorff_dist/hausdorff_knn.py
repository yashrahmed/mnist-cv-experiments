import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from bbox_extraction.extractor import extract_bbox_region
from common.contour_utils import get_contours, sample_points_from_contour
from common.dataset_utils import load_actual_mnist
from haussdorff_dist.distance import compute_hauss_dist


def cache_contours(images, idxs):
    cache = {}
    for i in range(1, images.shape[0] + 1):
        cache[idxs[i - 1]] = find_contours(images[i - 1])
    return cache


def find_contours(img):
    return sample_points_from_contour(get_contours(img)[0])


def img_haussdorff_distance_cached(train_contours, test_contours, ratio=0.2):
    def calculate(img_1_idx, img_2_idx):
        img_1_idx = img_1_idx[0]
        img_2_idx = img_2_idx[0]
        contour_1 = train_contours[img_1_idx] if img_1_idx > 0 else test_contours[img_1_idx]
        contour_2 = train_contours[img_2_idx] if img_2_idx > 0 else test_contours[img_2_idx]
        return compute_hauss_dist(contour_1, contour_2, ratio=ratio)
    return calculate


def pre_process_bbox(img):
    proc_img = extract_bbox_region(threshold_image(img))
    return proc_img


def run_knn(train_data, train_label, test_data, test_label, metric, neighbors=3, weights='distance', algo='brute',
            label='N/A'):
    knn = KNeighborsClassifier(n_neighbors=neighbors, metric=metric,
                               weights=weights, algorithm=algo)
    knn.fit(train_data, train_label)
    pred_labels = knn.predict(test_data)
    score = accuracy_score(test_label, pred_labels) * 100
    print(f'label={label}, N={neighbors}, score={score}%')


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

    #
    # train_images = train_images[0:n_train].reshape([-1, 28 * 28])
    # test_images = test_images[0:n_test].reshape([-1, 28 * 28])

    aligned_train_images = np.array([img for img in train_images[0:n_train]])
    aligned_test_images = np.array([img for img in test_images[0:n_test]])

    train_idxs = np.array([i for i in range(1, n_train + 1)], dtype=np.int32)
    test_idxs = np.array([-i for i in range(1, n_test + 1)], dtype=np.int32)

    train_cache = cache_contours(aligned_train_images, train_idxs)
    test_cache = cache_contours(aligned_test_images, test_idxs)
    dist_metric = img_haussdorff_distance_cached(train_cache, test_cache, ratio=0.25)
    run_knn(train_idxs.reshape([-1, 1]), train_labels[0:n_train], test_idxs.reshape([-1, 1]), test_labels[0:n_test],
            metric=dist_metric,
            label='Raw_Haussdorff')
