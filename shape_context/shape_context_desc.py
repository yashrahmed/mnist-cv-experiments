import numpy as np
from numpy import arctan2, histogram2d
from scipy.spatial.distance import pdist, squareform
from math import pi


def compute_shape_context_descriptor(vec, d_bin=5, t_bin=12):
    n, _ = vec.shape
    d_inner = 1
    d_outer = 40
    t_start = 0
    t_end = 2*pi
    d_bin_edges = np.logspace(np.log10(d_inner), np.log10(d_outer), d_bin)
    t_bin_edges = np.linspace(t_start, t_end, t_bin)

    dists = get_pairwise_dists(vec)
    angles = get_pairwise_slopes(vec)
    descs = np.array([histogram2d(dists[i, :], angles[i, :], bins=[d_bin_edges, t_bin_edges])[0] for i in range(n)])
    return descs.reshape([n, -1])


def get_pairwise_dists(vec):
    # vec is a NX2 array.
    n, _ = vec.shape
    dists = squareform(pdist(vec))
    return dists[np.where(dists > 0)].reshape([n, -1])


def get_pairwise_slopes(vec):
    n, _ = vec.shape
    xs = vec[:, 0:1]
    ys = vec[:, 1:2]
    dx = xs.transpose() - xs
    dy = ys.transpose() - ys
    angles = arctan2(dy, dx)
    return angles[np.where(np.abs(angles) > 0)].reshape([n, -1]) + pi
