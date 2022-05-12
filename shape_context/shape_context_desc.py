from math import pi

import numpy as np
from numpy import arctan2, histogram2d
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform


def calculate_correspondence(desc1, desc2, max_rank=20):
    cost_mat = compute_cost_matrix(desc1, desc2)
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    match_costs = np.array([cost_mat[row_ind[i]][col_ind[i]] for i in range(0, row_ind.shape[0])])
    cost_limit = np.max(np.partition(match_costs, max_rank)[:max_rank]) if match_costs.shape[0] > max_rank else np.max(
        match_costs)
    inlier_idxs = np.where(match_costs <= cost_limit)
    matches = np.vstack((row_ind, col_ind)).transpose()
    return matches, inlier_idxs, match_costs


def calculate_correspondence_for_manual_viz(desc1, desc2, max_rank=20):
    cost_mat = compute_cost_matrix(desc1, desc2)
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    match_costs = np.array([cost_mat[row_ind[i]][col_ind[i]] for i in range(0, row_ind.shape[0])])
    cost_limit = np.max(np.partition(match_costs, max_rank)[:max_rank]) if match_costs.shape[0] > max_rank else np.max(
        match_costs)
    inlier_idxs = np.where(match_costs <= cost_limit)
    matches = np.vstack((row_ind, col_ind)).transpose()
    return matches, inlier_idxs, match_costs, cost_mat, desc1, desc2


def compute_cost_matrix(desc1, desc2):
    r1, c1 = desc1.shape
    r2, c2 = desc2.shape
    desc1 = desc1.reshape([r1, 1, c1])
    desc2 = desc2.reshape([1, r2, c2])
    numr = np.power(desc1 - desc2, 2)
    denr = desc1 + desc2
    cost_mat = np.sum(np.divide(numr, np.where(denr > 0, denr, 0.01)), axis=2)
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
    d_inner = 0.01
    d_outer = 3
    t_start = 0
    t_end = 2 * pi + 0.01
    d_bin_edges = np.logspace(np.log10(d_inner), np.log10(d_outer), d_bin)
    t_bin_edges = np.linspace(t_start, t_end, t_bin)

    dists = get_pairwise_dists(vec)
    angles = get_pairwise_slopes(vec)
    descs = np.array([histogram2d(dists[i, :], angles[i, :], bins=[d_bin_edges, t_bin_edges])[0] for i in range(n)])
    return descs.reshape([n, -1])


def get_pairwise_dists(vec):
    # vec is a NX2 array.
    n, _ = vec.shape
    dists = squareform(pdist(vec.astype(np.float32)))
    return dists.reshape([n, -1]) / np.median(dists)


def get_pairwise_slopes(vec):
    n, _ = vec.shape
    xs = vec[:, 0:1].astype(np.float32)
    ys = vec[:, 1:2].astype(np.float32)
    dx = xs.transpose() - xs
    dy = ys.transpose() - ys
    angles = arctan2(dy, dx).reshape([n, -1]) + pi
    np.fill_diagonal(angles, -2)  # Fill diagonals with -2 so as to exclude from the histogram freq count.
    return angles
