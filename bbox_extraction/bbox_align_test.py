import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from bbox_extraction.bbox_extractor import extract_bbox_region
from common.dataset_utils import load_actual_mnist


def run_knn(train_data, train_label, test_data, test_label, neighbors=1, metric='euclidean', weights='distance',
            algo='auto'):
    n, r, c = train_data.shape
    knn = KNeighborsClassifier(n_neighbors=neighbors, metric=metric, weights=weights, algorithm=algo)
    knn.fit(train_data.reshape([-1, r * c]), train_label)
    pred_labels = knn.predict(test_data.reshape([-1, r * c]))
    score = accuracy_score(test_label, pred_labels) * 100
    print(f'N={neighbors},score={score}%')


if __name__ == '__main__':
    bbox_mode = 'align'
    train_images, train_labels, test_images, test_labels = load_actual_mnist()

    aligned_train_images = np.array([extract_bbox_region(img, mode=bbox_mode) for img in train_images])
    aligned_test_images = np.array([extract_bbox_region(img, mode=bbox_mode) for img in test_images])

    run_knn(aligned_train_images, train_labels, aligned_test_images, test_labels)
