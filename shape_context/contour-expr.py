import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from common.dataset_utils import load_actual_mnist
from common.plot_utils import scatter_plot

from shape_context_desc import compute_descriptor as get_sc


def threshold_image(images, threshold=100):
    return np.where(images > threshold, 255, 0).astype(np.uint8)


def run_clustering_on_image(image):
    x_points, y_points = np.where(image >= 255)
    points = np.vstack((x_points, y_points)).transpose().astype(np.uint16)

    agg = AgglomerativeClustering(n_clusters=10, linkage='average')
    agg.fit(points)
    return scatter_plot(points, agg.labels_)


def run_clustering_based_sampling_on_image(image, n_clusters=10):
    sample_points, uniq_labels = sample_points_using_clustering(image, n_clusters)
    plt = scatter_plot(sample_points, uniq_labels)
    return plt


def get_contours(bin_image):
    contours, hierarchy = cv2.findContours(bin_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy


def get_polygon(contour):
    epsilon = 0.01 * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


def get_polygons(contours):
    return [get_polygon(contour) for contour in contours]


def sample_points_using_clustering(image, n_clusters=10):
    x_points, y_points = np.where(image >= 255)
    points = np.vstack((x_points, y_points)).transpose().astype(np.uint16)

    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
    agg.fit(points)

    uniq_labels = np.unique(agg.labels_)
    sampled_points = np.array([np.mean(points[agg.labels_ == label], axis=0) for label in uniq_labels])
    return sampled_points, uniq_labels


if __name__ == '__main__':
    train_images, train_labels, _, _ = load_actual_mnist()
    image_of_6 = train_images[train_labels == 4][1914]
    image_of_6_col = cv2.cvtColor(image_of_6, cv2.COLOR_GRAY2BGR)
    th_image = threshold_image(image_of_6)
    # det_contours, _ = get_contours(th_image)
    # polygons = get_polygons(det_contours)
    # img = draw_polygons_on_image(image_of_6, polygons)
    run_clustering_on_image(th_image)
    plt_handle = run_clustering_based_sampling_on_image(th_image)
    plt_handle.show()
    # img = cv2.drawContours(image_of_6_col, det_contours, -1, (0, 0, 255), 1)
    # show_image(cv2.resize(img, (300, 300)))

    sample_points, _ = sample_points_using_clustering(th_image)
    descs = get_sc(sample_points)
    print(descs)
