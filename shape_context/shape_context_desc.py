from math import pi

import numpy as np
from numpy import arctan2, histogram2d
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import norm

def calculate_correspondence(desc1, desc2, max_rank=20):
    cost_mat = compute_cost_matrix(desc1, desc2)
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    match_costs = np.array([cost_mat[row_ind[i]][col_ind[i]] for i in range(0, row_ind.shape[0])])
    cost_limit = np.max(np.partition(match_costs, max_rank)[:max_rank]) if match_costs.shape[0] > max_rank else np.max(
        match_costs)
    inlier_idxs = np.where(match_costs <= cost_limit)
    matches = np.vstack((row_ind, col_ind)).transpose()
    return matches, inlier_idxs, match_costs


def calculate_correspondence_for_manual_viz(desc1, desc2, sample_pts_1, sample_pts_2, image_1, image_2):
    n1, _ = desc1.shape
    n2, _ = desc2.shape
    eps = 0.15  # Taken from the reference matlab implementation
    alpha = 0.7

    local_win_cost_mat = calculate_pairwise_local_window_dist(sample_pts_1, sample_pts_2, image_1, image_2)
    desc_cost_mat = compute_cost_matrix(desc1, desc2)
    cost_mat = alpha * desc_cost_mat + (1 - alpha) * local_win_cost_mat

    # Pad the cost matrix.
    cost_mat_size = max(n1, n2)
    cost_mat_padded = np.ones([cost_mat_size, cost_mat_size]) * eps
    cost_mat_padded[0:n1, 0:n2] = cost_mat

    # Perform matching.
    row_idxs, col_idxs = linear_sum_assignment(cost_mat_padded)

    # Points in desc1 that matched to one of the entries in desc2
    desc1_inliers_idxs = np.where(col_idxs[row_idxs[:n1]] < n2)
    # Points in desc2 that matched to one of the entries in desc1
    d2_idxs = np.argsort(col_idxs)[:n2]
    desc2_inliers_idxs = np.where(row_idxs[d2_idxs] < n1)

    matches = np.vstack((row_idxs, col_idxs)).transpose()[:n1]
    matches = matches[np.where(matches[:, 1] < n2)]
    match_costs = np.array([cost_mat_padded[matches[i][0]][matches[i][1]] for i in range(0, min(n1, n2))])
    hauss_eq_cost = calculate_hausdorff_eq_cost(cost_mat)
    return matches, desc1_inliers_idxs, desc2_inliers_idxs, match_costs, cost_mat, hauss_eq_cost, desc1, desc2


def calculate_hausdorff_eq_cost(cost_matrix):
    # Calculate shape context matching cost based on MATLAB reference implementation.
    return max(np.mean(np.min(cost_matrix, axis=0)), np.mean(np.min(cost_matrix, axis=1)))


def calculate_pairwise_local_window_dist(sample_pts_1, sample_pts_2, image_1, image_2):
    win_rad = 2
    h, w = image_1.shape
    # pad images
    pad_img_1 = np.zeros([h + win_rad, w + win_rad], dtype=np.uint8)
    pad_img_1[win_rad:win_rad + h, win_rad:win_rad + w] = image_1
    pad_img_2 = np.zeros([h + win_rad, w + win_rad], dtype=np.uint8)
    pad_img_2[win_rad:win_rad + h, win_rad:win_rad + w] = image_2

    # pad sample points
    sample_pts_1 = sample_pts_1 + [win_rad, win_rad]
    sample_pts_2 = sample_pts_2 + [win_rad, win_rad]

    distances = []
    for sample_1 in sample_pts_1:
        sample_dists = []
        x1, y1 = sample_1
        for sample_2 in sample_pts_2:
            x2, y2 = sample_2
            win_1 = pad_img_1[x1 - win_rad: x1 + win_rad, y1 - win_rad: y1 + win_rad]
            win_2 = pad_img_1[x2 - win_rad: x2 + win_rad, y2 - win_rad: y2 + win_rad]
            sample_dists.append(norm(win_1 - win_2))
        total_dist = np.sum(sample_dists)
        distances.append(np.array(sample_dists) / total_dist + 0.01)
    return np.array(distances)


def compute_cost_matrix(desc1, desc2):
    r1, c1 = desc1.shape
    r2, c2 = desc2.shape
    desc1 = desc1.reshape([r1, 1, c1])
    desc2 = desc2.reshape([1, r2, c2])
    numr = np.power(desc1 - desc2, 2)
    denr = desc1 + desc2
    cost_mat = np.sum(np.divide(numr, denr + 0.01), axis=2)
    return cost_mat / 2


# For debugging purposes only!
def compute_cost_matrix_raw(desc1, desc2):
    mat = []
    for r1 in desc1:
        costs = []
        for r2 in desc2:
            numr = np.power(r1 - r2, 2)
            denr = r1 + r2
            denr = np.where(denr > 0, denr, 0.01)
            costs.append(np.sum(np.divide(numr, denr)))
        mat.append(costs)
    return np.array(mat) / 2


def compute_descriptor(vec, d_bin=6, t_bin=13):
    n, _ = vec.shape
    vec = vec.astype(np.float32)
    # d_inner = 0.01
    # d_outer = 3
    # Using d_inner and d_outer values from reference MATLAB implementation.
    d_inner = 0.125
    d_outer = 2
    t_start = 0
    t_end = 2 * pi + 0.01
    d_bin_edges = np.logspace(np.log10(d_inner), np.log10(d_outer), d_bin)
    t_bin_edges = np.linspace(t_start, t_end, t_bin)

    dists = get_pairwise_dists(vec)
    angles = get_pairwise_slopes(vec)
    median_dist = np.median(dists)
    dists = dists / median_dist
    descs = []
    for i in range(n):
        histogram = histogram2d(dists[i, :], angles[i, :], bins=[d_bin_edges, t_bin_edges])[0]
        descs.append(histogram / (np.sum(histogram) + 0.01))
    return np.array(descs).reshape([n, -1]), median_dist


def get_pairwise_dists(vec):
    # vec is a NX2 array.
    n, _ = vec.shape
    dists = squareform(pdist(vec))
    return dists.reshape([n, -1])


def get_pairwise_slopes(vec):
    n, _ = vec.shape
    xs = vec[:, 0:1]
    ys = vec[:, 1:2]
    dx = xs.transpose() - xs
    dy = ys.transpose() - ys
    angles = arctan2(dy, dx).reshape([n, -1]) + pi
    np.fill_diagonal(angles, -2)  # Fill diagonals with -2 so as to exclude from the histogram freq count.
    return angles
