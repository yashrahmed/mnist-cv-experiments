from functools import reduce

import cv2
import numpy as np
from cv2.xfeatures2d import BriefDescriptorExtractor_create
from numpy.linalg import norm
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

from common.dataset_utils import load_dataset
from common.plot_utils import scatter_plot


def extract_brief(dataset, desc_size=16):
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

    images = dataset.get_attr('image')
    resized_imgs = [cv2.resize(img, resize_dims) for img in images]
    features = np.array([extractor.compute(img, fixed_keypoint)[1] for img in resized_imgs])
    return np.reshape(features, (images.shape[0], desc_size))


def euclidean_dist(dataset, ref_idx=0):
    images = dataset.get_attr('image')
    ref_image = images[[ref_idx]]
    ref_image_repeated = np.repeat(ref_image, images.shape[0], axis=0)
    return norm(ref_image_repeated - images, axis=(1, 2))


# flatten a N-d vector into a 2-d vector with the same # of rows
def flatten_inner(nd_vector):
    dims = list(nd_vector.shape)
    row = dims.pop(0)
    num_elems = reduce(lambda prod, i: prod * i, dims, 1)
    return nd_vector.reshape((row, num_elems))


def hamming_dist(dataset, ref_idx=0):
    vectors = dataset.get_attr('feature')
    n_rows = vectors.shape[0]
    ref_vector = vectors[[ref_idx]]
    ref_vector_repeated = np.repeat(ref_vector, n_rows, axis=0)
    return np.sum(np.unpackbits(ref_vector_repeated ^ vectors, axis=1), axis=1)


"""
############## MDS ################
"""


def compute_mds_dissimilarity(input_data):
    # input data is 2-d M*N array
    return np.sum(np.unpackbits(input_data[:, None, :] ^ input_data[None, :, :], axis=2), axis=2)


def run_mds_on_brief(dataset, n_comp=2):
    input_data = dataset.get_attr('feature')
    diss_matrix = compute_mds_dissimilarity(input_data)
    embedding = MDS(n_components=n_comp, dissimilarity='precomputed')
    return embedding.fit_transform(diss_matrix)


def run_mds_on_dataset(dataset, feat_key, n_comp=2):
    input_data = dataset.get_attr(feat_key)
    txfm_data = StandardScaler().fit_transform(flatten_inner(input_data))
    embedding = MDS(n_components=n_comp)
    return embedding.fit_transform(txfm_data)


"""
############## PCA ################
"""


def run_pca(input_data, n_comp=2):
    pca = PCA(n_components=n_comp)
    result = pca.fit_transform(input_data)
    return pca, result


def run_pca_on_dataset(dataset, feat_key, n_comp=2):
    input_data = dataset.get_attr(feat_key)
    txfm_data = StandardScaler().fit_transform(flatten_inner(input_data))
    return run_pca(txfm_data, n_comp=n_comp)


def scale_distances(distances, low=0, high=10):
    return np.interp(distances, (distances.min(), distances.max()), (low, high)).astype(np.uint8)


def _main():
    dataset = load_dataset()

    brief_features = extract_brief(dataset)
    dataset.add_attr('feature', brief_features)

    labels = dataset.get_attr('label')

    pca_raw_handle, pca_raw_results = run_pca_on_dataset(dataset, feat_key='image')
    print(f'pca on raw images -- {pca_raw_handle.explained_variance_ratio_ * 100}')
    scatter_plot(pca_raw_results, labels, 'PCA on Raw')

    pca_feat_handle, pca_feat_results = run_pca_on_dataset(dataset, feat_key='feature')
    print(f'pca on brief features -- {pca_feat_handle.explained_variance_ratio_ * 100}')
    plot_ref = scatter_plot(pca_feat_results, labels, 'PCA on BRIEF')

    mds_raw_results = run_mds_on_dataset(dataset, feat_key='image')
    scatter_plot(mds_raw_results, labels, 'MDS on Raw')

    mds_raw_results = run_mds_on_dataset(dataset, feat_key='feature')
    scatter_plot(mds_raw_results, labels, 'MDS on BRIEF (euclidean)')

    mds_feat_results = run_mds_on_brief(dataset)
    plot_ref = scatter_plot(mds_feat_results, labels, 'MDS on BRIEF (hamming**)')

    plot_ref.show()

    # loaded_images = dataset.get_attr('image')
    # show_images(loaded_images)


if __name__ == '__main__':
    _main()
