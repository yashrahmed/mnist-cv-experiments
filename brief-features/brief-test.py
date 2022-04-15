from functools import partial

import cv2
import numpy as np
from cv2.xfeatures2d import BriefDescriptorExtractor_create

from common.func_utils import compose_n
from common.img_utils import build_image_lookup, show_images


def create_brief_extractor_pipeline(desc_size=16):
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
    operations = [
        lambda img: cv2.resize(img, resize_dims),  # Resize image to a standard size
        lambda img: extractor.compute(img, fixed_keypoint)[1]  # extract feature for a single fixed keypoint
    ]
    return compose_n(operations)


def hamming_dist(v1, v2):
    return np.sum(np.unpackbits(v1 ^ v2))


if __name__ == '__main__':
    extract_brief = create_brief_extractor_pipeline()
    lookup = build_image_lookup()
    idxs = [(3, 41), (3, 45), (3, 23), (3, 34), (1, 1), (2, 21), (4, 11), (5, 15), (6, 56), (7, 36), (8, 25), (9, 44)]
    images = list(map(lookup, idxs))
    features = list(map(extract_brief, images))
    distances = list(map(partial(hamming_dist, features[0]), features))
    print('-------------- hamming distances --------------')
    print(distances)
    show_images(images)
