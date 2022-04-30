import cv2
import numpy as np


def show_images(images, disp_name='combined'):
    out_image = np.concatenate(images, axis=1)
    cv2.imshow(disp_name, out_image)
    cv2.waitKey(0)


def load_image(img_path):
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)


def plot_brief_features_on_image(image, coords, resize_value=(56, 56)):
    red_color = (0, 0, 255)
    green_color = (0, 255, 0)
    thickness = 1
    alpha = 0.5
    image = cv2.resize(image, resize_value)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = image.copy()
    for coord in coords:
        x1, y1, x2, y2 = coord
        overlay = cv2.line(overlay, (x1, y1), (x2, y2), green_color, thickness)
        overlay = cv2.rectangle(overlay, (x1, y1), (x1 + 1, y1 + 1), red_color, thickness)
        overlay = cv2.rectangle(overlay, (x2, y2), (x2 + 1, y2 + 1), red_color, thickness)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
