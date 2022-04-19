from functools import reduce

import cv2
import numpy as np
from cv2.xfeatures2d import BriefDescriptorExtractor_create
from numpy.linalg import norm
from numpy.random import seed
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

from common.dataset_utils import load_dataset
from common.plot_utils import scatter_plot


def extract_brief(images, desc_size=16):
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


def scale_distances(distances, low=0, high=10):
    return np.interp(distances, (distances.min(), distances.max()), (low, high)).astype(np.uint8)


def scale_feature(input_array):
    return StandardScaler().fit_transform(input_array)


def unpack_bits(input_array):
    return np.unpackbits(input_array, axis=1)


"""
############## MDS ################
"""


def compute_mds_dissimilarity(input_data):
    # input data is 2-d M*N array
    return np.sum(np.unpackbits(input_data[:, None, :] ^ input_data[None, :, :], axis=2), axis=2)


def run_mds_hamming(input_data, n_comp=2):
    diss_matrix = compute_mds_dissimilarity(input_data)
    embedding = MDS(n_components=n_comp, dissimilarity='precomputed')
    return embedding.fit_transform(diss_matrix)


def run_mds_euclidean(input_data, n_comp=2):
    embedding = MDS(n_components=n_comp)
    return embedding.fit_transform(input_data)


"""
############## PCA ################
"""


def run_pca(input_data, n_comp=2):
    pca = PCA(n_components=n_comp)
    result = pca.fit_transform(input_data)
    return pca, result


"""
############## Feature Selection ################
"""


def select_with_variance_threshold(feature_data, p=0.8):
    """
    Boolean features are Bernoulli random variables, and the variance of such variables is given by
    var(X) = p * (1 - p)
    The goal of variance threshold selection is to remove features that are only 0's or 1's more than p% of the time.
    """
    sel = VarianceThreshold(threshold=(p * (1 - p)))
    selected_features = sel.fit_transform(np.unpackbits(feature_data, axis=1))
    return np.packbits(selected_features, axis=1)


def select_with_k_best(feature_data, labels, num_features=10):
    selected_features = SelectKBest(chi2, k=num_features).fit_transform(np.unpackbits(feature_data, axis=1), labels)
    return np.packbits(selected_features, axis=1)


"""
############## Main ################
"""


def run_mds_experiments(**kwargs):
    images = kwargs.get('image')
    labels = kwargs.get('labels')
    brief_features = kwargs.get('brief_features')
    pruned_features = kwargs.get('pruned_features')
    k_best_features = kwargs.get('k_best_features')

    mds_raw_results = run_mds_euclidean(scale_feature(flatten_inner(images)))
    scatter_plot(mds_raw_results, labels, 'MDS on Raw')

    mds_feat_results = run_mds_hamming(brief_features)
    scatter_plot(mds_feat_results, labels, 'MDS on BRIEF (hamming**)')

    mds_feat_results = run_mds_hamming(pruned_features)
    scatter_plot(mds_feat_results, labels, 'Variance Threshold')

    mds_feat_results = run_mds_hamming(k_best_features)
    return scatter_plot(mds_feat_results, labels, 'K best selected')


def run_pca_experiments(**kwargs):
    images = kwargs.get('image')
    labels = kwargs.get('labels')
    brief_features = kwargs.get('brief_features')
    pruned_features = kwargs.get('pruned_features')
    k_best_features = kwargs.get('k_best_features')

    pca_raw_handle, pca_raw_results = run_pca(scale_feature(flatten_inner(images)))
    print(f'pca on raw images -- {pca_raw_handle.explained_variance_ratio_ * 100}')
    scatter_plot(pca_raw_results, labels, 'PCA on Raw')

    pca_ref, pca_feat_results = run_pca(scale_feature(unpack_bits(brief_features)))
    print(f'pca on brief features -- {pca_ref.explained_variance_ratio_ * 100}')
    scatter_plot(pca_feat_results, labels, 'PCA on BRIEF')

    pca_ref, pca_feat_results = run_pca(scale_feature(unpack_bits(pruned_features)))
    print(f'pca on selected brief features -- {pca_ref.explained_variance_ratio_ * 100}')
    scatter_plot(pca_feat_results, labels, f'PCA on BRIEF (With Variance Threshold Selection)')

    pca_ref, pca_feat_results = run_pca(scale_feature(unpack_bits(k_best_features)))
    print(f'pca on selected brief features -- {pca_ref.explained_variance_ratio_ * 100}')
    return scatter_plot(pca_feat_results, labels, f'PCA on BRIEF (With K best feature select - Chi2)')


def _main():
    dataset = load_dataset()
    images = dataset.get_attr('image')
    labels = dataset.get_attr('label')
    brief_features = extract_brief(images)
    pruned_features = select_with_variance_threshold(brief_features, p=0.65)
    k_best_features = select_with_k_best(brief_features, labels, num_features=60)

    plot_ref = run_pca_experiments(image=images, labels=labels, brief_features=brief_features,
                                   pruned_features=pruned_features,
                                   k_best_features=k_best_features)

    plot_ref = run_mds_experiments(image=images, labels=labels, brief_features=brief_features,
                                   pruned_features=pruned_features,
                                   k_best_features=k_best_features)

    plot_ref.show()

    cov_mat = np.cov(unpack_bits(brief_features), rowvar=False)
    print('------- Covariance matrix of brief features. -------')
    print(cov_mat)

    # loaded_images = dataset.get_attr('image')
    # show_images(loaded_images)


if __name__ == '__main__':
    seed(0)
    _main()
