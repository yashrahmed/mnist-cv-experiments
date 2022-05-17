import numpy as np
from numpy.linalg import norm


def compute_hauss_dist(pts_1, pts_2, ratio=0.6):
    r1, c1 = pts_1.shape
    r2, c2 = pts_2.shape

    assert c1 == c2 == 2
    assert 0 < ratio <= 1

    p1_kth_eq_count = int(ratio * (r1 - 1))
    p2_kth_eq_count = int(ratio * (r2 - 1))

    # Reshape for broadcasting
    pts_1 = pts_1.astype(np.float32).reshape([r1, 1, 2])
    pts_2 = pts_2.astype(np.float32).reshape([1, r2, 2])

    dist_mat = norm(pts_1 - pts_2, axis=2)

    maxmin_along_row = np.sort(np.min(dist_mat, axis=1))[-1::-1][p1_kth_eq_count]
    maxmin_along_col = np.sort(np.min(dist_mat, axis=0))[-1::-1][p2_kth_eq_count]

    return max(maxmin_along_row, maxmin_along_col)
