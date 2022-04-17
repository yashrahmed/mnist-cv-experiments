import cv2
import numpy as np
from cv2.xfeatures2d import BriefDescriptorExtractor_create
from numpy.linalg import norm

from common.utils import load_dataset, show_images


def extract_brief(dataframe, desc_size=16):
    """
        OpenCV indirectly imposes a size restriction.
        See https://github.com/opencv/opencv_contrib/blob/342f8924cca88fe6ce979024b7776f6815c89978/modules/xfeatures2d/src/brief.cpp#L249
        //Remove keypoints very close to the border
        KERNEL_SIZE=48 and PATCH_SIZE=9
        KeyPointsFilter::runByImageBorder(keypoints, image.size(), PATCH_SIZE/2 + KERNEL_SIZE/2);
        Also see https://github.com/opencv/opencv_contrib/blob/master/modules/xfeatures2d/src/generated_32.i
        1. OpenCV does not do pixel comparison. Instead it compares differences of cumulative gray values
           b/w small image patches.
        2. Integral image are used for #1.
        3. OpenCV supports 16, 32 and 64 BYTE BRIEF descriptors. Each value in the descriptor array is an 8-bit signature.
        4. Encoding each value in the descriptor to binary and concatenating will give the complete descriptor.
    """
    resize_dims = (60, 60)
    fixed_keypoint = [cv2.KeyPoint(30, 30, 0)]  # A single Keypoint fixed at the center of the image.
    extractor = BriefDescriptorExtractor_create(desc_size)

    images = dataframe['image'].values
    resized_imgs = [cv2.resize(img, resize_dims) for img in images.flat]
    return [extractor.compute(img, fixed_keypoint)[1] for img in resized_imgs]


def euclidean_dist(dataframe, ref_idx=0):
    images = dataframe['image'][idxs].values
    ref_image = images[ref_idx]
    return np.array([norm(ref_image - img) for img in images.flat])


def hamming_dist(dataframe, ref_idx=0):
    features = dataframe['features'].values
    vectors = np.array([x[0] for x in features.flat])
    n_rows = vectors.shape[0]
    ref_vector = vectors[[ref_idx]]
    ref_vector_repeated = np.repeat(ref_vector, n_rows, axis=0)
    return np.sum(np.unpackbits(ref_vector_repeated ^ vectors, axis=1), axis=1)


def scale_distances(distances, low=0, high=10):
    return np.interp(distances, (distances.min(), distances.max()), (low, high)).astype(np.uint8)


if __name__ == '__main__':
    dataset = load_dataset()
    idxs = [210, 105, 55, 551]  # @ToDo figure out lookup
    subset = dataset.iloc[idxs].copy()

    brief_features = extract_brief(subset)
    subset['features'] = brief_features

    feature_ham_distances = hamming_dist(subset)
    img_euc_distance = euclidean_dist(subset)
    print('-------------- Hamming distances b/w feature --------------')
    print(scale_distances(feature_ham_distances))
    print('-------------- Euclidean distance b/w feature --------------')
    print(scale_distances(img_euc_distance))

    loaded_images = subset['image'][idxs].values
    show_images(loaded_images)
