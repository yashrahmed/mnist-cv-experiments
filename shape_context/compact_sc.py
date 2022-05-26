import numpy as np
from numpy import histogram2d


def compute_compact_sc(points, bbox, d_bin=5):
    x_min, y_min, x_max, y_max = bbox

    x_row_scaled = np.interp(points[:, 0], (x_min, x_max), (0, 1))
    y_row_scaled = np.interp(points[:, 1], (y_min, y_max), (0, 1))

    d_inner = 0
    d_outer = 1.01
    d_bin_edges = np.linspace(d_inner, d_outer, d_bin + 1)

    histogram = histogram2d(x_row_scaled, y_row_scaled, bins=[d_bin_edges, d_bin_edges])[0]

    return histogram / (np.sum(histogram) + 0.01)
