import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from common.dataset_utils import load_actual_mnist
from common.plot_utils import scatter_plot
from common.img_utils import plot_matches, show_image
from shape_context_desc import compute_descriptor as get_sc, compute_cost_matrix, calculate_correspondence


def threshold_image(images, threshold=100):
    return np.where(images > threshold, 255, 0).astype(np.uint8)


def run_clustering_on_image(image):
    x_points, y_points = np.where(image >= 255)
    points = np.vstack((x_points, y_points)).transpose().astype(np.uint16)

    agg = AgglomerativeClustering(n_clusters=10, linkage='average')
    agg.fit(points)
    return scatter_plot(points, agg.labels_)


def run_clustering_based_sampling_on_image(image, n_clusters=10, title='title'):
    sample_points = sample_points_using_clustering(image, n_clusters)
    plt = scatter_plot(sample_points, np.arange(0, n_clusters), title=title)
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

    unique_labels = np.unique(agg.labels_)
    sampled_points = np.array([np.mean(points[agg.labels_ == label], axis=0) for label in unique_labels])
    return sampled_points


def run_on_control_images_expr():
    center1 = (12, 12)
    center2 = (20, 20)
    control_4_corners = np.zeros([28, 28], dtype=np.uint8)
    control_4_corners[center1[0]-5:center1[0]-5+2, center1[1]-5:center1[1]-5+2] = 255
    control_4_corners[center1[0]-5:center1[0]-5+2, center1[1]+5:center1[1]+5+2] = 255
    control_4_corners[center1[0]+5:center1[0]+5+2, center1[1]-5:center1[1]-5+2] = 255
    control_4_corners[center1[0]+5:center1[0]+5+2, center1[1]+5:center1[1]+5+2] = 255

    control_4_corners_2 = np.zeros([28, 28], dtype=np.uint8)
    control_4_corners_2[center2[0]-5:center2[0]-5+2, center2[1]-5:center2[1]-5+2] = 255
    control_4_corners_2[center2[0]-5:center2[0]-5+2, center2[1]+5:center2[1]+5+2] = 255
    control_4_corners_2[center2[0]+5:center2[0]+5+2, center2[1]-5:center2[1]-5+2] = 255
    control_4_corners_2[center2[0]+5:center2[0]+5+2, center2[1]+5:center2[1]+5+2] = 255

    control_diamond = np.zeros([28, 28], dtype=np.uint8)
    control_diamond[center1[0]:center1[0]+2, center1[1]-5:center1[1]-5+2] = 255
    control_diamond[center1[0]:center1[0]+2, center1[1]+5:center1[1]+5+2] = 255
    control_diamond[center1[0]-5:center1[0]-5+2, center1[1]:center1[1]+2] = 255
    control_diamond[center1[0]+5:center1[0]+5+2, center1[1]:center1[1]+2] = 255

    sp_c = sample_points_using_clustering(control_4_corners, n_clusters=4)
    descs_c = get_sc(sp_c)

    sp_c2 = sample_points_using_clustering(control_4_corners_2, n_clusters=4)
    descs_c2 = get_sc(sp_c2)

    sp_d = sample_points_using_clustering(control_diamond, n_clusters=4)
    descs_d = get_sc(sp_d)

    mat = compute_cost_matrix(descs_c, descs_c2)
    mat2 = compute_cost_matrix(descs_c, descs_d)

    matches, total_cost = calculate_correspondence(mat)
    show_image(plot_matches(control_4_corners, control_4_corners_2, sp_c, sp_c2, matches))
    matches, total_cost_2 = calculate_correspondence(mat2)
    show_image(plot_matches(control_4_corners, control_diamond, sp_c, sp_d, matches))
    print('histograms of corner image #1 ------------------->')
    print(descs_c.reshape([4, 5, 12]))
    print('histograms of corner image #2 ------------------->')
    print(descs_c2.reshape([4, 5, 12]))
    print('histograms of diamonds image ------------------->')
    print(descs_d.reshape([4, 5, 12]))

    print(f'cost of matching corner to corner_2 = {total_cost}')
    print(f'cost of matching corner to diamond = {total_cost_2}')


if __name__ == '__main__':
    train_images, train_labels, _, _ = load_actual_mnist()
    image_of_4 = threshold_image(train_images[train_labels == 4][1914])
    image_of_42 = threshold_image(train_images[train_labels == 4][194])
    image_of_5 = threshold_image(train_images[train_labels == 5][719])

    # det_contours, _ = get_contours(image_of_4)
    # polygons = get_polygons(det_contours)
    # img = draw_polygons_on_image(image_of_4, polygons)
    # run_clustering_on_image(image_of_5)
    # plt_handle = run_clustering_based_sampling_on_image(image_of_4, title='4')
    # plt_handle = run_clustering_based_sampling_on_image(image_of_42, title='42')
    # plt_handle = run_clustering_based_sampling_on_image(image_of_5, title='5')
    # plt_handle.show()
    # img = cv2.drawContours(image_of_6_col, det_contours, -1, (0, 0, 255), 1)
    # show_image(cv2.resize(img, (300, 300)))

    sp_4 = sample_points_using_clustering(image_of_4)
    descs_4 = get_sc(sp_4)

    sp_42 = sample_points_using_clustering(image_of_42)
    descs_42 = get_sc(sp_42)

    sp_5 = sample_points_using_clustering(image_of_5)
    descs_5 = get_sc(sp_5)
    #
    mat = compute_cost_matrix(descs_4, descs_42)
    mat2 = compute_cost_matrix(descs_4, descs_5)

    matches, total_cost = calculate_correspondence(mat)
    show_image(plot_matches(image_of_4, image_of_42, sp_4, sp_42, matches))
    print(total_cost)
    print('*******************')
    matches, total_cost = calculate_correspondence(mat2)
    show_image(plot_matches(image_of_4, image_of_5, sp_4, sp_5, matches))
    print(total_cost)
