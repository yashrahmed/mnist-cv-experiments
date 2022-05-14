import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from common.dataset_utils import load_actual_mnist


def threshold_image(image, th=70):
    return cv2.threshold(image, th, 255, cv2.THRESH_BINARY)[1]


def get_contours(image):
    th_img = threshold_image(image)
    contours, _ = cv2.findContours(th_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def get_bbox(image):
    pts_x, pts_y = np.where(image == 255)
    # bbox = cv2.boundingRect(np.vstack((pts_y, pts_x)).transpose()) # Bug in OpenCV version.
    return np.min(pts_x), np.min(pts_y), np.max(pts_x), np.max(pts_y)


def extract_bbox(image, b_box):
    x1, y1, x2, y2 = b_box
    return cv2.resize(image[x1:x2, y1:y2], (28, 28))


def bbox_alignment(image):
    b_box = get_bbox(threshold_image(image))
    return extract_bbox(image, b_box)


def run_knn(train_data, train_label, test_data, test_label, neighbors=1, metric='euclidean', weights='distance',
            algo='auto'):
    knn = KNeighborsClassifier(n_neighbors=neighbors, metric=metric, weights=weights, algorithm=algo)
    knn.fit(train_data.reshape([-1, 28 * 28]), train_label)
    pred_labels = knn.predict(test_data.reshape([-1, 28 * 28]))
    score = accuracy_score(test_label, pred_labels) * 100
    print(f'N={neighbors},score={score}%')


if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_actual_mnist()
    img = train_images[12475]

    aligned_train_images = np.array([bbox_alignment(img) for img in train_images])
    aligned_test_images = np.array([bbox_alignment(img) for img in test_images])

    run_knn(aligned_train_images, train_labels, aligned_test_images, test_labels)

    # contours = get_contours(img)
    # bbox = get_bbox(threshold_image(img))
    #
    # img2 = to_color(img)
    # draw_rects_on_image(img2, [bbox])
    # show_image(img2)
    #
    # show_image(extract_bbox(img, bbox))

    # for i in range(0, train_images.shape[0]):
    #     contours = get_contours(train_images[i])
    #     if len(contours) > 1:
    #         print(f'index={i} label={train_labels[i]}')
