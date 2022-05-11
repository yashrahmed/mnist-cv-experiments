import cv2
import numpy as np
from numpy.linalg import norm
from scipy.interpolate import RBFInterpolator as RBF
from sklearn.cluster import AgglomerativeClustering
from tps import ThinPlateSpline

from common.dataset_utils import load_actual_mnist
from common.img_utils import draw_matches, show_image, show_images
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
    contours, hierarchy = cv2.findContours(bin_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return contours, hierarchy


def get_polygon(contour):
    epsilon = 0.01 * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


def get_polygons(contours):
    return [get_polygon(contour) for contour in contours]


def morph(point_matches, pts_1, pts_2, img_1):
    r, c = img_1.shape
    out_img = np.zeros([r, c]).astype(np.uint8)
    n, _ = point_matches.shape
    matched_pts = np.array([pts_2[i, :] for i in point_matches[:n, 1]]).reshape([-1, 2])
    interp = RBF(pts_1[:n, :], matched_pts, kernel='linear', neighbors=8)
    x_points, y_points = np.where(img_1 >= 255)
    points = np.vstack((x_points, y_points)).transpose().astype(np.uint8)
    new_points = np.round(interp(points)).astype(np.uint8)
    pts_1 = np.round(interp(pts_1)).astype(np.uint8)
    for pt in new_points:
        if 0 <= pt[0] < r and 0 <= pt[1] < c:
            out_img[pt[0]][pt[1]] = 255
    return out_img, pts_1


def morph_pure_tps(point_matches, pts_1, pts_2, img_1):
    r, c = img_1.shape
    out_img = np.zeros([r, c]).astype(np.uint8)
    n, _ = point_matches.shape
    matched_pts = np.array([pts_2[i, :] for i in point_matches[:n, 1]]).reshape([-1, 2])
    tps = ThinPlateSpline(0.01)
    tps.fit(pts_1[:n, :], matched_pts)
    x_points, y_points = np.where(img_1 >= 255)
    points = np.vstack((x_points, y_points)).transpose().astype(np.uint8)
    new_points = np.round(tps.transform(points)).astype(np.uint8)
    pts_1 = np.round(tps.transform(pts_1)).astype(np.uint8)
    for pt in new_points:
        if 0 <= pt[0] < r and 0 <= pt[1] < c:
            out_img[pt[0]][pt[1]] = 255
    return out_img, pts_1


def morph_affine(point_matches, pts_1, pts_2, img_1):
    pts_1 = swap_cols(pts_1)
    pts_2 = swap_cols(pts_2)
    n, _ = point_matches.shape
    matched_pts = np.array([pts_2[i, :] for i in point_matches[:n, 1]])
    mat, mask = cv2.estimateAffine2D(pts_1[:n, :], matched_pts, method=cv2.RANSAC, ransacReprojThreshold=3)
    new_img = threshold_image(cv2.warpAffine(img_1, mat, img_1.shape), 70)
    # swap to bring back to numpy axes convention.
    new_pts = swap_cols(np.reshape(cv2.transform(pts_1.reshape([-1, 1, 2]), mat), [-1, 2]))
    return new_img, new_pts


def morph_homo(point_matches, pts_1, pts_2, img_1):
    pts_1 = swap_cols(pts_1)
    pts_2 = swap_cols(pts_2)
    n, _ = point_matches.shape
    matched_pts = np.array([pts_2[i, :] for i in point_matches[:n, 1]])
    mat, mask = cv2.findHomography(pts_1[:n, :], matched_pts, cv2.RANSAC, 6)
    return threshold_image(cv2.warpPerspective(img_1, mat, img_1.shape), 80)


def morph_points(point_matches, pts_1, pts_2, img_1):
    n, _ = point_matches.shape
    matched_pts = np.array([pts_2[i, :] for i in point_matches[:n, 1]]).reshape([-1, 2])
    interp = RBF(pts_1[:n, :], matched_pts, smoothing=0.01)
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
    return swap_cols(np.unique(np.vstack(contours).astype(np.uint16).reshape([-1, 2]), axis=0))


def run_on_control_images_expr():
    control_4_corners, control_4_corners_2, control_diamond = create_control_shapes()

    sp_c = sample_points_using_clustering(control_4_corners, n_clusters=4)
    descs_c = get_sc(sp_c)

    sp_c2 = sample_points_using_clustering(control_4_corners_2, n_clusters=4)
    descs_c2 = get_sc(sp_c2)

    sp_d = sample_points_using_clustering(control_diamond, n_clusters=4)
    descs_d = get_sc(sp_d)

    matches, _, total_cost = calculate_correspondence(descs_c, descs_c2)
    show_image(draw_matches(control_4_corners, control_4_corners_2, sp_c, sp_c2, matches))
    matches, _, total_cost_2 = calculate_correspondence(descs_c, descs_d)
    show_image(draw_matches(control_4_corners, control_diamond, sp_c, sp_d, matches))
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

    matches, _, total_cost_first_time = calculate_correspondence(descs_1, descs_2)
    show_image(draw_matches(image_1, image_2, sp_1, sp_2, matches))

    total_cost_after_final_morph = total_cost_first_time
    # Perform morphing k times.....
    for i in range(0, k):
        image_1 = morph(matches, sp_1, sp_2, image_1)
        sp_1 = sample_points_using_clustering(image_1, n_clusters=n_clusters)
        descs_1 = get_sc(sp_1)

        matches, _, total_cost_after_final_morph = calculate_correspondence(descs_1, descs_2)
        show_image(draw_matches(image_1, image_2, sp_1, sp_2, matches))

    print(f'first time cost = {total_cost_first_time}')
    print(f'{k}th time cost = {total_cost_after_final_morph}')


def run_contour_sc_distance_with_morph(image_1, image_2, viz=True):
    # image 1 and 2 are binary images.
    contour_1, _ = get_contours(image_1)
    sp_1 = sample_points_from_contour(contour_1)
    descs_1 = get_sc(sp_1)

    contour_2, _ = get_contours(image_2)
    sp_2 = sample_points_from_contour(contour_2)
    descs_2 = get_sc(sp_2)

    matches, inlier_matches, total_cost_first_time = calculate_correspondence(descs_1, descs_2, max_rank=250)

    if viz:
        show_image(draw_matches(image_1, image_2, sp_1, sp_2, inlier_matches))

    # Morph once.......
    image_1, sp_1 = morph_pure_tps(inlier_matches, sp_1, sp_2, image_1)
    diff = norm(image_1 - image_2)

    if viz:
        print(f'image distance after morphing = {diff}')
        # image_1 = draw_points_on_image(image_1, sp_1)
        # image_2 = draw_points_on_image(image_2, sp_2)
        show_images([image_1, image_2], scale=10)

    return diff


def run_sc_distance_with_morph_with_homography(image_1, image_2, k=1, n_clusters=20):
    # image 1 and 2 are binary images.
    sp_1 = sample_points_using_clustering(image_1, n_clusters=n_clusters)
    descs_1 = get_sc(sp_1)

    sp_2 = sample_points_using_clustering(image_2, n_clusters=n_clusters)
    descs_2 = get_sc(sp_2)

    matches, _, total_cost_first_time = calculate_correspondence(descs_1, descs_2)
    show_image(draw_matches(image_1, image_2, sp_1, sp_2, matches))

    total_cost_after_final_morph = total_cost_first_time
    # Perform morphing k times.....
    for i in range(0, k):
        image_1 = morph_homo(matches, sp_1, sp_2, image_1)
        sp_1 = sample_points_using_clustering(image_1, n_clusters=n_clusters)
        descs_1 = get_sc(sp_1)

        matches, _, total_cost_after_final_morph = calculate_correspondence(descs_1, descs_2)
        show_image(draw_matches(image_1, image_2, sp_1, sp_2, matches))

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


def run_batch_scoring_experiments(images, labels):
    # Results might be a little misleading as they may not directly apply to a KNN application.
    img_of_4 = threshold_image(images[labels == 4][314])
    tgt_match_images = images[labels == 4][0:50]
    tgt_non_match_images = images[labels == 8][0:50]
    sc_match_count = 0
    trivial_match_count = 0
    total_count = 0
    for i, m_img in enumerate(tgt_match_images):
        m_img = threshold_image(m_img)
        for j, nm_img in enumerate(tgt_non_match_images):
            # print(i,j)
            nm_img = threshold_image(nm_img)
            tgt_match_dist = run_contour_sc_distance_with_morph(m_img, img_of_4, viz=False)
            other_match_dist = run_contour_sc_distance_with_morph(nm_img, img_of_4, viz=False)
            tgt_match_dist_simple = norm(img_of_4 - m_img)
            other_match_dist_match_dist_simple = norm(img_of_4 - nm_img)
            if tgt_match_dist < other_match_dist:
                sc_match_count += 1
            if tgt_match_dist_simple < other_match_dist_match_dist_simple:
                trivial_match_count += 1
            total_count += 1
    print(
        f'With SC: Correctly discriminated in {sc_match_count} of {total_count} trials. {(sc_match_count / total_count) * 100}%')
    print(
        f'With Simple: Correctly discriminated in {trivial_match_count} of {total_count} trials. {(trivial_match_count / total_count) * 100}%')


if __name__ == '__main__':
    train_images, train_labels, _, _ = load_actual_mnist()
    image_of_4 = threshold_image(train_images[train_labels == 4][314])

    image_of_42 = threshold_image(train_images[train_labels == 4][10])
    image_of_5 = threshold_image(train_images[train_labels == 5][0])
    run_contour_sc_distance_with_morph(image_of_42, image_of_4)
    run_contour_sc_distance_with_morph(image_of_5, image_of_4)
