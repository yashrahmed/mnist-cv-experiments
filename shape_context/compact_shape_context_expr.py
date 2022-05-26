import numpy as np

from bbox_extraction.bbox_extractor import get_bbox_of_points
from common.dataset_utils import load_actual_mnist
from common.img_utils import threshold_image, draw_rects_on_image, to_color, show_image, render_desc
from shape_context.compact_sc import compute_compact_sc


def build_compact_sc(image, d_bin=15):
    idx = np.where(threshold_image(image, 80) == 255)
    points = np.vstack((idx[0], idx[1])).transpose()
    bbox = get_bbox_of_points(points)
    x_min, y_min, x_max, y_max = bbox
    sc = compute_compact_sc(points, bbox, d_bin)

    out_img = to_color(image)
    draw_rects_on_image(out_img, [[x_min, y_min, x_max, y_max]])
    show_image(out_img)
    show_image(render_desc(sc, 'compact-sc', d_bin, d_bin))
    return sc


def main():
    train_images, train_labels, test_images, test_labels = load_actual_mnist()
    sc = build_compact_sc(train_images[54712])


if __name__ == '__main__':
    main()
