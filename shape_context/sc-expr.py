import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator as RBF
from sklearn.cluster import AgglomerativeClustering

from common.dataset_utils import load_actual_mnist
from common.img_utils import plot_matches, show_image, draw_contours_on_image
from common.plot_utils import scatter_plot
from shape_context_desc import compute_descriptor as get_sc, calculate_correspondence


def swap_cols(x):
    return np.array(x[:, [1, 0]])


def create_control_shapes():
    center1 = (12, 12)
    center2 = (20, 20)
    control_4_corners = np.zeros([28, 28], dtype=np.uint8)
    control_4_corners[center1[0] - 5:center1[0] - 5 + 2, center1[1] - 5:center1[1] - 5 + 2] = 255
    control_4_corners[center1[0] - 5:center1[0] - 5 + 2, center1[1] + 5:center1[1] + 5 + 2] = 255
    control_4_corners[center1[0] + 5:center1[0] + 5 + 2, center1[1] - 5:center1[1] - 5 + 2] = 255
    control_4_corners[center1[0] + 5:center1[0] + 5 + 2, center1[1] + 5:center1[1] + 5 + 2] = 255

    control_4_corners_2 = np.zeros([28, 28], dtype=np.uint8)
    control_4_corners_2[center2[0] - 5:center2[0] - 5 + 2, center2[1] - 5:center2[1] - 5 + 2] = 255
    control_4_corners_2[center2[0] - 5:center2[0] - 5 + 2, center2[1] + 5:center2[1] + 5 + 2] = 255
    control_4_corners_2[center2[0] + 5:center2[0] + 5 + 2, center2[1] - 5:center2[1] - 5 + 2] = 255
    control_4_corners_2[center2[0] + 5:center2[0] + 5 + 2, center2[1] + 5:center2[1] + 5 + 2] = 255

    control_diamond = np.zeros([28, 28], dtype=np.uint8)
    control_diamond[center1[0]:center1[0] + 2, center1[1] - 5:center1[1] - 5 + 2] = 255
    control_diamond[center1[0]:center1[0] + 2, center1[1] + 5:center1[1] + 5 + 2] = 255
    control_diamond[center1[0] - 5:center1[0] - 5 + 2, center1[1]:center1[1] + 2] = 255
    control_diamond[center1[0] + 5:center1[0] + 5 + 2, center1[1]:center1[1] + 2] = 255

    return control_4_corners, control_4_corners_2, control_diamond


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
    contours, hierarchy = cv2.findContours(bin_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy


def get_polygon(contour):
    epsilon = 0.01 * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


def get_polygons(contours):
    return [get_polygon(contour) for contour in contours]


def morph(point_matches, pts_1, pts_2, img_1):
    out_img = np.zeros([28, 28]).astype(np.uint8)
    n1, _ = pts_1.shape
    n2, _ = pts_2.shape
    n = min(n1, n2)
    matched_pts = np.array([pts_2[i, :] for i in point_matches[:n, 1]]).reshape([-1, 2])
    interp = RBF(pts_1[:n, :], matched_pts, kernel='linear')
    x_points, y_points = np.where(img_1 >= 255)
    points = np.vstack((x_points, y_points)).transpose().astype(np.uint8)
    new_points = np.round(interp(points)).astype(np.uint8)
    for pt in new_points:
        out_img[pt[0]][pt[1]] = 255
    return out_img


def morph_homo(point_matches, pts_1, pts_2, img_1):
    pts_1 = swap_cols(pts_1)
    pts_2 = swap_cols(pts_2)
    n1, _ = pts_1.shape
    n2, _ = pts_2.shape
    n = min(n1, n2)
    matched_pts = np.array([pts_2[i, :] for i in point_matches[:n, 1]]).reshape([-1, 1, 2])
    mat, mask = cv2.findHomography(pts_1[:n, :].reshape([-1, 1, 2]), matched_pts, cv2.RANSAC, 6)
    return threshold_image(cv2.warpPerspective(img_1, mat, img_1.shape), 80)


def morph_points(point_matches, pts_1, pts_2, img_1):
    n1, _ = pts_1.shape
    n2, _ = pts_2.shape
    n = min(n1, n2)
    matched_pts = np.array([pts_2[i, :] for i in point_matches[:n, 1]]).reshape([-1, 2])
    interp = RBF(pts_1[:n, :], matched_pts, kernel='linear')
    x_points, y_points = np.where(img_1 >= 255)
    points = np.vstack((x_points, y_points)).transpose().astype(np.uint8)
    new_points = np.round(interp(points)).astype(np.uint8)
    return new_points


def sample_points_using_clustering(image, n_clusters=10):
    x_points, y_points = np.where(image >= 90)
    points = np.vstack((x_points, y_points)).transpose().astype(np.uint16)

    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
    agg.fit(points)

    unique_labels = np.unique(agg.labels_)
    sampled_points = np.array([np.mean(points[agg.labels_ == label], axis=0) for label in unique_labels])
    return sampled_points


def sample_points_from_contour(contours):
    # Assumes that the input is a simplified contour.
    return swap_cols(np.unique(np.vstack(contours).astype(np.uint8).reshape([-1, 2]), axis=0))


def run_on_control_images_expr():
    control_4_corners, control_4_corners_2, control_diamond = create_control_shapes()

    sp_c = sample_points_using_clustering(control_4_corners, n_clusters=4)
    descs_c = get_sc(sp_c)

    sp_c2 = sample_points_using_clustering(control_4_corners_2, n_clusters=4)
    descs_c2 = get_sc(sp_c2)

    sp_d = sample_points_using_clustering(control_diamond, n_clusters=4)
    descs_d = get_sc(sp_d)

    matches, total_cost = calculate_correspondence(descs_c, descs_c2)
    show_image(plot_matches(control_4_corners, control_4_corners_2, sp_c, sp_c2, matches))
    matches, total_cost_2 = calculate_correspondence(descs_c, descs_d)
    show_image(plot_matches(control_4_corners, control_diamond, sp_c, sp_d, matches))
    print('histograms of corner image #1 ------------------->')
    print(descs_c.reshape([4, 5, 12]))
    print('histograms of corner image #2 ------------------->')
    print(descs_c2.reshape([4, 5, 12]))
    print('histograms of diamonds image ------------------->')
    print(descs_d.reshape([4, 5, 12]))

    print(f'cost of matching corner to corner_2 = {total_cost}')
    print(f'cost of matching corner to diamond = {total_cost_2}')


def run_sc_distance_with_morph(image_1, image_2, k=1, n_clusters=30):
    # image 1 and 2 are binary images.
    sp_1 = sample_points_using_clustering(image_1, n_clusters=n_clusters)
    descs_1 = get_sc(sp_1)

    sp_2 = sample_points_using_clustering(image_2, n_clusters=n_clusters)
    descs_2 = get_sc(sp_2)

    matches, total_cost_first_time = calculate_correspondence(descs_1, descs_2)
    show_image(plot_matches(image_1, image_2, sp_1, sp_2, matches))

    total_cost_after_final_morph = total_cost_first_time
    # Perform morphing k times.....
    for i in range(0, k):
        image_1 = morph(matches, sp_1, sp_2, image_1)
        sp_1 = sample_points_using_clustering(image_1, n_clusters=n_clusters)
        descs_1 = get_sc(sp_1)

        matches, total_cost_after_final_morph = calculate_correspondence(descs_1, descs_2)
        show_image(plot_matches(image_1, image_2, sp_1, sp_2, matches))

    print(f'first time cost = {total_cost_first_time}')
    print(f'{k}th time cost = {total_cost_after_final_morph}')


def run_contour_sc_distance_with_morph(image_1, image_2, k=1):
    # image 1 and 2 are binary images.
    contour_1, _ = get_contours(image_1)
    sp_1 = sample_points_from_contour(contour_1)
    descs_1 = get_sc(sp_1)

    contour_2, _ = get_contours(image_2)
    sp_2 = sample_points_from_contour(contour_2)
    descs_2 = get_sc(sp_2)

    matches, total_cost_first_time = calculate_correspondence(descs_1, descs_2)
    show_image(plot_matches(image_1, image_2, sp_1, sp_2, matches))

    total_cost_after_final_morph = total_cost_first_time
    # Perform morphing k times.....
    for i in range(0, k):
        image_1 = morph_homo(matches, sp_1, sp_2, image_1)
        contour_1, _ = get_contours(image_1)
        sp_1 = sample_points_from_contour(contour_1)
        descs_1 = get_sc(sp_1)

        matches, total_cost_after_final_morph = calculate_correspondence(descs_1, descs_2)
        show_image(draw_contours_on_image(plot_matches(image_1, image_2, sp_1, sp_2, matches), contour_1))

    print(f'first time cost = {total_cost_first_time}')
    print(f'{k}th time cost = {total_cost_after_final_morph}')


def run_sc_distance_with_morph_with_homography(image_1, image_2, k=1, n_clusters=20):
    # image 1 and 2 are binary images.
    sp_1 = sample_points_using_clustering(image_1, n_clusters=n_clusters)
    descs_1 = get_sc(sp_1)

    sp_2 = sample_points_using_clustering(image_2, n_clusters=n_clusters)
    descs_2 = get_sc(sp_2)

    matches, total_cost_first_time = calculate_correspondence(descs_1, descs_2)
    show_image(plot_matches(image_1, image_2, sp_1, sp_2, matches))

    total_cost_after_final_morph = total_cost_first_time
    # Perform morphing k times.....
    for i in range(0, k):
        image_1 = morph_homo(matches, sp_1, sp_2, image_1)
        sp_1 = sample_points_using_clustering(image_1, n_clusters=n_clusters)
        descs_1 = get_sc(sp_1)

        matches, total_cost_after_final_morph = calculate_correspondence(descs_1, descs_2)
        show_image(plot_matches(image_1, image_2, sp_1, sp_2, matches))

    print(f'first time cost = {total_cost_first_time}')
    print(f'{k}th time cost = {total_cost_after_final_morph}')


def builtin_shape_context_dist_experiment():
    # Fails to run. Exit code 139 with SIGSEGV (Seg fault)
    control_4_corners, control_4_corners_2, control_diamond = create_control_shapes()

    sp_c = sample_points_using_clustering(control_4_corners, n_clusters=4)
    sp_c2 = sample_points_using_clustering(control_4_corners_2, n_clusters=4)
    sp_d = sample_points_using_clustering(control_diamond, n_clusters=4)

    dist_extractor = cv2.ShapeContextDistanceExtractor()
    print(
        f'dist b/w corner shapes = {dist_extractor.computeDistance(sp_c.reshape([4, 1, 2]), sp_c2.reshape([4, 1, 2]))}')


if __name__ == '__main__':
    train_images, train_labels, _, _ = load_actual_mnist()
    image_of_4 = threshold_image(train_images[train_labels == 4][314])
    image_of_42 = threshold_image(train_images[train_labels == 5][64])
    image_of_5 = threshold_image(train_images[train_labels == 4][1120])

    run_contour_sc_distance_with_morph(image_of_4, image_of_42, k=1)
    run_contour_sc_distance_with_morph(image_of_4, image_of_5, k=1)

