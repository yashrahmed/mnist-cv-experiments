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

    # Inverting the distance (i.e. multiplying by -1) and the formula evaluation allows the uses of partitioning
    # instead of sorting.
    dist_mat *= -1
    max_min_along_row_opt = np.partition(np.max(dist_mat, axis=1), p1_kth_eq_count)[p1_kth_eq_count]
    max_min_along_col_opt = np.partition(np.max(dist_mat, axis=0), p2_kth_eq_count)[p2_kth_eq_count]
    distance = min(max_min_along_row_opt, max_min_along_col_opt) * -1

    """
        # The operations below simulate how OpenCV calculates the Kth max!
        # which is incorrect!!!! Simply picking the value in the Kth row without
        # sorting yeilds incorrect results. The hausdorff distance requires calculating
        # the Kth ranked point.
        maxmin_along_row = np.min(dist_mat, axis=1)[p1_kth_eq_count]
        maxmin_along_col = np.min(dist_mat, axis=0)[p2_kth_eq_count]
    """
    return distance
