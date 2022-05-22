import cv2
import numpy as np

from common.contour_utils import get_contours_from_image
from common.dataset_utils import load_actual_mnist
from common.img_utils import threshold_image


def create_matcher(ref_contours, ref_labels):
    dist_extractor = cv2.createShapeContextDistanceExtractor()

    def matcher(contour, label):
        min_idx = np.argmin(
            [dist_extractor.computeDistance(contour, ref_contours[i]) for i in range(ref_labels.shape[0])])
        return label == ref_labels[min_idx]

    return matcher


def generate_samples(images, labels, samples_per_class=3):
    digits = np.unique(labels)
    sampled_images = None
    for digit in digits:
        samples = images[np.where(labels == digit)][0:samples_per_class]
        sampled_images = samples if sampled_images is None else np.concatenate((sampled_images, samples), axis=0)
    return sampled_images, np.repeat(digits, samples_per_class)


def extract_contours(images, th=90):
    return [get_contours_from_image(threshold_image(img, th), opencv_fmt=True) for img in images]


def run_matches(matcher, query_images, query_gt_labels):
    return np.sum([matcher(query_images[i], query_gt_labels[i]) for i in range(query_gt_labels.shape[0])])


def main():
    n_test = 500
    train_images, train_labels, test_images, test_labels = load_actual_mnist()
    sampled_images, sampled_labels = generate_samples(train_images, train_labels, 15)
    contour_for_samples = extract_contours(sampled_images)
    contour_for_test_set = extract_contours(test_images[:n_test])
    sc_matcher = create_matcher(contour_for_samples, sampled_labels)
    match_count = run_matches(sc_matcher, contour_for_test_set, test_labels[:n_test])
    print(f'Score = {match_count / n_test * 100}%')


if __name__ == '__main__':
    main()
