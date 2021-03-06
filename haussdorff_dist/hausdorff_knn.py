import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from bbox_extraction.bbox_extractor import extract_bbox_region
from common.contour_utils import get_contours, sample_points_from_contour
from common.dataset_utils import load_actual_mnist
from haussdorff_dist.distance import compute_hauss_dist
from haussdorff_dist.mnist_exemplar import generate_exemplar_dataset


def cache_contours(images, idxs, opencv_fmt=False):
    cache = {}
    for i in range(1, images.shape[0] + 1):
        cache[idxs[i - 1]] = find_contours(images[i - 1], opencv_fmt)
    return cache


def find_contours(img, opencv_fmt):
    return sample_points_from_contour(get_contours(img, cv2.CHAIN_APPROX_NONE)[0], opencv_fmt, dtype=np.float32)


def img_haussdorff_distance_cached(train_contours, test_contours, ratio=0.2):
    def calculate(img_1_idx, img_2_idx):
        img_1_idx = img_1_idx[0]
        img_2_idx = img_2_idx[0]
        contour_1 = train_contours[img_1_idx] if img_1_idx > 0 else test_contours[img_1_idx]
        contour_2 = train_contours[img_2_idx] if img_2_idx > 0 else test_contours[img_2_idx]
        return compute_hauss_dist(contour_1, contour_2, ratio=ratio)

    return calculate


def img_sc2_distance_cached(train_contours, test_contours):
    extractor = cv2.createShapeContextDistanceExtractor()

    # requires contours in opencv format [N, 1, 2]
    def calculate(img_1_idx, img_2_idx):
        img_1_idx = img_1_idx[0]
        img_2_idx = img_2_idx[0]
        contour_1 = train_contours[img_1_idx] if img_1_idx > 0 else test_contours[img_1_idx]
        contour_2 = train_contours[img_2_idx] if img_2_idx > 0 else test_contours[img_2_idx]
        return extractor.computeDistance(contour_1, contour_2)

    return calculate


def pre_process_bbox(img, th=70, mode='extract'):
    return extract_bbox_region(img, th, mode=mode)


def run_knn(train_data, train_label, test_data, test_label, metric, neighbors=3, weights='distance', algo='brute',
            label='N/A'):
    knn = KNeighborsClassifier(n_neighbors=neighbors, metric=metric,
                               weights=weights, algorithm=algo)
    knn.fit(train_data, train_label)
    pred_labels = knn.predict(test_data)
    score = accuracy_score(test_label, pred_labels) * 100
    print(f'label={label}, N={neighbors}, score={score}%')


if __name__ == '__main__':
    pre_process_th = 70
    exemplar_th = 150
    n_clusters_per_label = 50
    n_train = 5000  # n_clusters_per_label * 10
    n_test = 500
    mode = 'extract'

    train_images, train_labels, test_images, test_labels = load_actual_mnist()

    aligned_train_images = np.array([pre_process_bbox(img, pre_process_th, mode=mode) for img in train_images])
    aligned_test_images = np.array([pre_process_bbox(img, pre_process_th, mode=mode) for img in test_images])

    # exemplar_train_images, exemplar_labels = generate_exemplar_dataset(aligned_train_images, train_labels,
    #                                                                    clusters_per_label=n_clusters_per_label)
    # exemplar_train_images = np.array([threshold_image(img, exemplar_th) for img in exemplar_train_images])

    train_idxs = np.array([i for i in range(1, train_labels.shape[0] + 1)], dtype=np.int32)
    test_idxs = np.array([-i for i in range(1, test_labels.shape[0] + 1)], dtype=np.int32)

    # Prepare final input after preprocessing
    input_train_images = aligned_train_images[0:n_train]
    input_train_idxs = train_idxs[0:n_train]
    input_train_labels = train_labels[0:n_train]

    input_test_images = aligned_test_images[0:n_test]
    input_test_idxs = test_idxs[0:n_test]
    input_test_labels = test_labels[0:n_test]

    train_contours_cache = cache_contours(input_train_images, input_train_idxs)
    test_contours_cache = cache_contours(input_test_images, input_test_idxs)
    dist_metric = img_haussdorff_distance_cached(train_contours_cache, test_contours_cache, ratio=0.25)
    run_knn(input_train_idxs.reshape([-1, 1]), input_train_labels, input_test_idxs.reshape([-1, 1]), input_test_labels,
            metric=dist_metric,
            label='Aligned_Haussdorff')
