import cv2
import numpy as np
from scipy.spatial import KDTree


def draw_rects_on_image(image, rects):
    assert len(image.shape) == 3  # Ensure that the input is a 3 channel image.
    for rectangle in rects:
        y1, x1, y2, x2 = rectangle
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)


def draw_contours_on_image(image, contours):
    assert len(image.shape) == 3  # Ensure that the input is a 3 channel image.
    cv2.drawContours(image, contours, -1, color=(0, 255, 0))


def draw_polygons_on_image(image, polygons):
    assert len(image.shape) == 3  # Ensure that the input is a 3 channel image.
    for polygon in polygons:
        cv2.polylines(image, [polygon], True, (0, 255, 0), 1)


def draw_points_on_image(image, points):
    assert len(image.shape) == 3  # Ensure that the input is a 3 channel image.
    w, h, _ = image.shape
    scale = 5
    image = cv2.resize(image, (w * scale, h * scale))
    for point in np.round(points) * scale:
        cv2.rectangle(image, (point[1] - 2, point[0] - 2), (point[1] + 2, point[0] + 2), (0, 255, 0), 1)
    return image


def draw_brief_features_on_image(image, coords, resize_value=(56, 56)):
    assert len(image.shape) == 3  # Ensure that the input is a 3 channel image.
    red_color = (0, 0, 255)
    green_color = (0, 255, 0)
    cyan_color = (255, 10, 0)
    thickness = 1
    alpha = 0.5
    image = cv2.resize(image, resize_value)
    overlay = image.copy()
    for coord in coords:
        # Swap columns to reflect OpenCV conventions.
        y1, x1, y2, x2 = coord
        cv2.line(overlay, (x1, y1), (x2, y2), green_color, thickness)
        cv2.rectangle(overlay, (x1, y1), (x1 + 1, y1 + 1), red_color, thickness)
        cv2.rectangle(overlay, (x2, y2), (x2 + 1, y2 + 1), cyan_color, thickness)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def draw_matches(img_1, img_2, points_1, points_2, matches):
    assert len(img_1.shape) == 3  # Ensure that the input is a 3 channel image.
    assert len(img_2.shape) == 3  # Ensure that the input is a 3 channel image.
    assert img_1.shape == img_2.shape
    red_color = (0, 0, 255)
    blue_color = (255, 0, 0)
    scale = 5
    w, h, _ = img_1.shape
    new_size = (w * scale, h * scale)
    thickness = 1
    img_1 = cv2.resize(img_1, new_size)
    img_2 = cv2.resize(img_2, new_size)
    img_3 = cv2.hconcat([img_1, img_2])
    points_1 = np.round(points_1) * scale
    points_2 = (np.round(points_2) + [0, w]) * scale  # offset to account for a concatenated image
    for point in points_1:
        y, x = point
        cv2.rectangle(img_3, (x, y), (x + 5, y + 5), blue_color, thickness)
    for point in points_2:
        y, x = point
        cv2.rectangle(img_3, (x, y), (x + 5, y + 5), blue_color, thickness)
    for match in matches:
        m1, m2 = match
        y1, x1 = points_1[m1]
        y2, x2 = points_2[m2]
        cv2.line(img_3, (x1, y1), (x2, y2), red_color, thickness)
    return img_3


def draw_matches_for_manual_viz(img_1, img_2, points_1, points_2, matches, costs, cost_mat, desc1, desc2):
    def invert(tup):
        return tup[1], tup[0]

    def find_bbox(points):
        left = np.min(points[:, 1])
        right = np.max(points[:, 1])
        top = np.min(points[:, 0])
        bottom = np.max(points[:, 0])
        return (top, left), (bottom, right)

    def scale_to_bbox(bbox, points):
        (top, left), (bottom, right) = bbox
        out_points = np.copy(points).astype(np.float32)
        out_points[:, 0] = (points[:, 0] - top) / (bottom - top + 0.001)
        out_points[:, 1] = (points[:, 1] - left) / (right - left + 0.001)
        return out_points

    def get_nns(scaled_points_query, scaled_points_tgt):
        kd_tree = KDTree(scaled_points_tgt, leafsize=5)
        _, idxs = kd_tree.query(scaled_points_query)
        return idxs

    def render_descs(tgt_desc, match_desc, best_cost_desc, nn_desc):
        desc_img_top = cv2.hconcat((render_desc(tgt_desc, 'tgt_desc'), render_desc(match_desc, 'matched_desc')))
        desc_img_bottom = cv2.hconcat((render_desc(best_cost_desc, 'low_cost_desc'), render_desc(nn_desc, 'NN_desc')))
        return cv2.vconcat((desc_img_top, desc_img_bottom))

    assert len(img_1.shape) == 3  # Ensure that the input is a 3 channel image.
    assert len(img_2.shape) == 3  # Ensure that the input is a 3 channel image.
    assert img_1.shape == img_2.shape

    red_color = (0, 0, 255)
    blue_color = (255, 0, 0)
    yellow_color = (0, 255, 255)
    green_color = (0, 255, 0)
    cyan_color = (255, 255, 0)
    pink_color = (203, 192, 255)

    assert img_1.shape == img_2.shape
    scale = 5
    w, h, _ = img_1.shape
    new_size = (w * scale, h * scale)
    thickness = 1
    img_1 = cv2.resize(img_1, new_size)
    img_2 = cv2.resize(img_2, new_size)
    img_3 = cv2.hconcat([img_1, img_2])

    points_1 = np.round(points_1) * scale
    points_2 = (np.round(points_2) + [0, w]) * scale  # offset to account for a concatenated image

    # Calculate and draw first bounding box
    top_left, bottom_right = find_bbox(points_1)
    cv2.rectangle(img_3, invert(top_left), invert(bottom_right), green_color, thickness)
    scaled_pts_1 = scale_to_bbox((top_left, bottom_right), points_1)

    # Calculate and draw second bounding box
    top_left, bottom_right = find_bbox(points_2)
    cv2.rectangle(img_3, invert(top_left), invert(bottom_right), green_color, thickness)
    scaled_pts_2 = scale_to_bbox((top_left, bottom_right), points_2)

    # Nearest neighbor matching
    nn_idxs = get_nns(scaled_pts_1, scaled_pts_2)
    points_1_nn = points_2[nn_idxs]
    nn_descs = desc2[nn_idxs]

    # Draw points.
    for point in points_1:
        y, x = point
        cv2.rectangle(img_3, (x - 2, y - 2), (x + 2, y + 2), blue_color, thickness)
    for point in points_2:
        y, x = point
        cv2.rectangle(img_3, (x - 2, y - 2), (x + 2, y + 2), blue_color, thickness)
    for i, match in enumerate(matches):
        m1, m2 = match
        y1, x1 = points_1[m1]
        y2, x2 = points_2[m2]

        img_match = np.copy(img_3)

        # Draw actual match a
        cv2.line(img_match, (x1, y1), (x2, y2), red_color, thickness)

        # Draw local windows
        win_rad = 12  # local window radius
        cv2.rectangle(img_match, (x1 - win_rad, y1 - win_rad), (x1 + win_rad, y1 + win_rad), pink_color, thickness)
        cv2.rectangle(img_match, (x2 - win_rad, y2 - win_rad), (x2 + win_rad, y2 + win_rad), pink_color, thickness)

        top_3_idx = np.argsort(cost_mat[m1])[:3]

        # Draw top-3 matches
        # for idx in top_3_idx:
        #     y2_idx, x2_idx = points_2[idx]
        #     cv2.line(img_match, (x1, y1), (x2_idx, y2_idx), green_color, thickness)

        # Draw best match by cost
        bes_match_idx = top_3_idx[0]
        y2_best, x2_best = points_2[bes_match_idx]
        img_match = cv2.line(img_match, (x1, y1), (x2_best, y2_best), yellow_color, thickness)

        # Draw best match by NN search on position
        y2_nn, x2_nn = points_1_nn[m1]
        img_match = cv2.line(img_match, (x1, y1), (x2_nn, y2_nn), cyan_color, thickness)
        print(f'{x1},{y1} --> {x2},{y2} : LA_cost={costs[i]} LOW_cost={cost_mat[m1][bes_match_idx]}')

        # Render descriptors
        cv2.imshow('descriptors', render_descs(desc1[m1], desc2[m2], desc2[bes_match_idx], nn_descs[m1]))
        cv2.imshow('Matches', img_match)
        cv2.waitKey(0)


def load_image(img_path):
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)


def render_desc(desc, label, d_bin=5, t_bin=12):
    magenta_color = (255, 0, 255)
    lowest = np.min(desc)
    highest = np.max(desc)
    pixels_per_desc = 20

    n_row_desc_arr = d_bin * pixels_per_desc
    n_col_desc_arr = t_bin * pixels_per_desc
    desc_array = desc.reshape([d_bin, t_bin]).repeat(pixels_per_desc, axis=0).repeat(pixels_per_desc, axis=1)
    desc_img = np.zeros([n_row_desc_arr + 20, n_col_desc_arr]).astype(np.uint8)
    desc_img = cv2.cvtColor(desc_img, cv2.COLOR_GRAY2BGR)
    desc_img[:n_row_desc_arr, :, 1] = np.interp(desc_array, [lowest, highest], [15, 255]).astype(np.uint8)
    desc_img[:n_row_desc_arr, :, 2] = desc_img[:n_row_desc_arr, :, 1]
    cv2.putText(desc_img, label, (n_row_desc_arr, n_col_desc_arr), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                color=magenta_color)
    return desc_img


def show_image(image, disp_name='single'):
    cv2.imshow(disp_name, image)
    cv2.waitKey(0)


def show_images(images, disp_name='combined', scale=1):
    shape = images[0].shape
    r = shape[0]
    c = shape[1]
    if not scale == 1:
        images = [cv2.resize(image, (r * scale, c * scale)) for image in images]
    out_image = np.concatenate(images, axis=1)
    cv2.imshow(disp_name, out_image)
    cv2.waitKey(0)


def threshold_image(image, th=70):
    return cv2.threshold(image, th, 255, cv2.THRESH_BINARY)[1]


def to_color(image):
    ch = len(image.shape)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if ch == 2 else image
