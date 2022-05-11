import cv2
import numpy as np


def draw_contours_on_image(image, contours):
    return cv2.drawContours(image, contours, -1, color=(0, 255, 0))


def draw_polygons_on_image(image, polygons):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for polygon in polygons:
        cv2.polylines(image, [polygon], True, (0, 255, 0), 1)
    return image


def draw_points_on_image(image, points):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for point in points:
        cv2.drawMarker(image, (point[1], point[0]), (0, 255, 0), cv2.MARKER_SQUARE, markerSize=1, thickness=1)
    return image


def draw_brief_features_on_image(image, coords, resize_value=(56, 56)):
    red_color = (0, 0, 255)
    green_color = (0, 255, 0)
    thickness = 1
    alpha = 0.5
    image = cv2.resize(image, resize_value)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = image.copy()
    for coord in coords:
        x1, y1, x2, y2 = coord
        cv2.line(overlay, (x1, y1), (x2, y2), green_color, thickness)
        cv2.rectangle(overlay, (x1, y1), (x1 + 1, y1 + 1), red_color, thickness)
        cv2.rectangle(overlay, (x2, y2), (x2 + 1, y2 + 1), red_color, thickness)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def draw_matches(img_1, img_2, points_1, points_2, matches):
    red_color = (0, 0, 255)
    blue_color = (255, 0, 0)
    new_size = (280, 280)
    thickness = 1
    img_1 = cv2.cvtColor(cv2.resize(img_1, new_size), cv2.COLOR_GRAY2BGR)
    img_2 = cv2.cvtColor(cv2.resize(img_2, new_size), cv2.COLOR_GRAY2BGR)
    img_3 = cv2.hconcat([img_1, img_2])
    points_1 = np.round(points_1) * 10
    points_2 = (np.round(points_2) + [0, 28]) * 10  # offset to account for a concatenated image
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


def draw_matches_for_manual_viz(img_1, img_2, points_1, points_2, matches, costs, inlier_idxs, cost_mat, desc1, desc2):
    red_color = (0, 0, 255)
    blue_color = (255, 0, 0)
    yellow_color = (0, 255, 255)
    green_color = (255, 0, 0)
    new_size = (280, 280)
    thickness = 1
    img_1 = cv2.cvtColor(cv2.resize(img_1, new_size), cv2.COLOR_GRAY2BGR)
    img_2 = cv2.cvtColor(cv2.resize(img_2, new_size), cv2.COLOR_GRAY2BGR)
    img_3 = cv2.hconcat([img_1, img_2])
    points_1 = np.round(points_1) * 10
    points_2 = (np.round(points_2) + [0, 28]) * 10  # offset to account for a concatenated image
    for point in points_1:
        y, x = point
        cv2.rectangle(img_3, (x, y), (x + 5, y + 5), blue_color, thickness)
    for point in points_2:
        y, x = point
        cv2.rectangle(img_3, (x, y), (x + 5, y + 5), blue_color, thickness)
    for i, match in enumerate(matches):
        m1, m2 = match
        y1, x1 = points_1[m1]
        y2, x2 = points_2[m2]
        top_3_idx = np.argsort(cost_mat[m1])[:3]
        nn_idx = top_3_idx[0]
        img_match = np.copy(img_3)
        cv2.line(img_match, (x1, y1), (x2, y2), red_color, thickness)
        for idx in top_3_idx:
            y2_idx, x2_idx = points_2[idx]
            cv2.line(img_match, (x1, y1), (x2_idx, y2_idx), green_color, thickness)
        y2_best, x2_best = points_2[nn_idx]
        img_match = cv2.line(img_match, (x1, y1), (x2_best, y2_best), yellow_color, thickness)
        print(f'{x1},{y1} --> {x2},{y2} : LA_cost={costs[i]} NN_cost={cost_mat[m1][nn_idx]}')
        cv2.imshow('yo', img_match)
        cv2.waitKey(0)


def load_image(img_path):
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)


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
