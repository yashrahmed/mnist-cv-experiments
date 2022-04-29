from functools import reduce

import cv2
import numpy as np
from cv2.xfeatures2d import BriefDescriptorExtractor_create
from numpy.linalg import norm
from numpy.random import seed
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.manifold import MDS
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common.dataset_utils import load_actual_mnist
from common.img_utils import show_images, plot_brief_features_on_image
from common.plot_utils import scatter_plot
from parse_brief_coords import load_coords

SEED = 0


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
############## KNN ################
"""


def split_dataset(images, labels, test_ratio=0.3):
    return train_test_split(images, labels, test_size=test_ratio, random_state=SEED)


def run_knn(train_data, train_label, test_data, test_label, neighbors=4, metric='euclidean', weights='uniform',
            algo='auto'):
    knn = KNeighborsClassifier(n_neighbors=neighbors, metric=metric, weights=weights, algorithm=algo)
    knn.fit(train_data, train_label)
    pred_labels = knn.predict(test_data)
    return accuracy_score(test_label, pred_labels) * 100


def run_knn_experiment(neighbors, brief_desc_size, test_split_ratio, weights, flat_img_train, label_train,
                       flat_img_test,
                       label_test, feat_train, feat_test):
    baseline_acc = run_knn(flat_img_train, label_train, flat_img_train, label_train, neighbors=neighbors,
                           weights=weights)
    raw_img_acc = run_knn(flat_img_train, label_train, flat_img_test, label_test, neighbors=neighbors, weights=weights)
    feat_acc = run_knn(feat_train, label_train, feat_test, label_test, neighbors=neighbors, metric='hamming',
                       weights=weights)
    gain = feat_acc - raw_img_acc
    print(f'{test_split_ratio},{weights},{neighbors},{brief_desc_size},{baseline_acc},{raw_img_acc},{feat_acc},{gain}')


def run_knn_experiment_set(neighbors_values, weights_values, brief_desc_size_values, images, labels):
    test_split_ratio = 0.3
    img_train, img_test, label_train, label_test = split_dataset(images, labels, test_ratio=test_split_ratio)
    img_train = flatten_inner(img_train)  # standard scaling makes it worse
    img_test = flatten_inner(img_test)
    print('test_split_ratio,weights,neighbors,brief_desc_size,baseline_acc,raw_img_acc,feat_acc,gain')
    for brief_desc_size in brief_desc_size_values:
        brief_features = unpack_bits(extract_brief(images, desc_size=brief_desc_size))
        feat_train, feat_test, _, _ = split_dataset(brief_features, labels, test_ratio=test_split_ratio)
        for neighbors in neighbors_values:
            for weights in weights_values:
                run_knn_experiment(neighbors, brief_desc_size, test_split_ratio, weights, img_train, label_train,
                                   img_test,
                                   label_test, feat_train, feat_test)


def run_knn_on_full_mnist_raw_experiment(n_neighbors=3):
    train_images, train_labels, test_images, test_labels = load_actual_mnist()
    accuracy = run_knn(flatten_inner(train_images), train_labels, flatten_inner(test_images), test_labels,
                       neighbors=n_neighbors, weights='distance')
    print(f'Classification accuracy for full dataset with raw images @ K={n_neighbors} is {accuracy}')


def run_knn_on_full_mnist_brief_experiment(n_neighbors=3, desc_size=16):
    train_images, train_labels, test_images, test_labels = load_actual_mnist()
    train_brief = extract_brief(train_images, desc_size=desc_size)
    test_brief = extract_brief(test_images, desc_size=desc_size)
    accuracy = run_knn(unpack_bits(train_brief), train_labels, unpack_bits(test_brief), test_labels,
                       neighbors=n_neighbors, metric='hamming',
                       weights='distance')
    print(
        f'Classification accuracy for full dataset using brief features of {desc_size} bytes @ K={n_neighbors} is {accuracy}')


def run_knn_on_full_mnist_experiment(neighbors_values):
    for neighbors in neighbors_values:
        print(f'with {neighbors}NN ----------')
        run_knn_on_full_mnist_raw_experiment(n_neighbors=neighbors)
        run_knn_on_full_mnist_brief_experiment(n_neighbors=neighbors, desc_size=16)
        run_knn_on_full_mnist_brief_experiment(n_neighbors=neighbors, desc_size=32)
        run_knn_on_full_mnist_brief_experiment(n_neighbors=neighbors, desc_size=64)


def run_knn_on_full_mnist_avg_images_brief_experiment(n_neighbors=3, desc_size=16):
    train_images, train_labels, test_images, test_labels = load_actual_mnist()
    avg_train_images, train_labels = generate_average_dataset(train_images, train_labels)
    train_brief = extract_brief(avg_train_images, desc_size=desc_size)
    test_brief = extract_brief(test_images, desc_size=desc_size)
    accuracy = run_knn(unpack_bits(train_brief), train_labels, unpack_bits(test_brief), test_labels,
                       neighbors=n_neighbors, metric='hamming',
                       weights='distance')
    print(
        f'Classification accuracy for full dataset using brief features of {desc_size} on average images bytes @ K={n_neighbors} is {accuracy}')


"""
############## Decision Tree ################
"""


def run_decision_tree_with_brief_experiment():
    train_images, train_labels, test_images, test_labels = load_actual_mnist()
    tree = DecisionTreeClassifier()
    for desc_size in [16, 32, 64]:
        train_brief = unpack_bits(extract_brief(train_images, desc_size=desc_size))
        test_brief = unpack_bits(extract_brief(test_images, desc_size=desc_size))
        tree.fit(train_brief, train_labels)
        pred_labels = tree.predict(test_brief)
        score = accuracy_score(test_labels, pred_labels) * 100
        print(f'Classification accuracy with a decision tree on {desc_size}-byte brief features = {score}%')


"""
############## Support Vector Machines ################
"""


def run_svm_with_brief_experiment(kernel='linear'):
    train_images, train_labels, test_images, test_labels = load_actual_mnist()
    svm = SVC(max_iter=5000, kernel=kernel)
    for desc_size in [16, 32, 64]:
        train_brief = np.interp(unpack_bits(extract_brief(train_images, desc_size=desc_size)), [0, 1], [-1, 1])
        test_brief = np.interp(unpack_bits(extract_brief(test_images, desc_size=desc_size)), [0, 1], [-1, 1])
        svm.fit(train_brief, train_labels)
        pred_labels = svm.predict(test_brief)
        score = accuracy_score(test_labels, pred_labels) * 100
        print(f'Classification accuracy with a svm on {desc_size}-byte brief features = {score}%')


def run_svm_with_raw_images_experiment():
    train_images, train_labels, test_images, test_labels = load_actual_mnist()
    svm = SVC(max_iter=5000)
    train_brief = scale_feature(flatten_inner(train_images))
    test_brief = scale_feature(flatten_inner(test_images))
    svm.fit(train_brief, train_labels)
    pred_labels = svm.predict(test_brief)
    score = accuracy_score(test_labels, pred_labels) * 100
    print(f'Classification accuracy with a svm on raw images = {score}%')


"""
############## IMAGE AVERAGING ################
"""


def generate_average_dataset(images, labels):
    digits = np.unique(labels)
    average_images = []
    for digit in digits:
        mean_img = np.mean(images[labels == digit, :, :], axis=0).astype(np.uint8)
        average_images.append(mean_img)
    return np.array(average_images), digits


def run_image_averaging_experiment(images, labels):
    avg_imgs, _ = generate_average_dataset(images, labels)
    show_images(avg_imgs)


"""
############## Main ################
"""


def run_covariance_inspect_experiment(brief_features):
    cov_mat = np.cov(unpack_bits(brief_features), rowvar=False)
    print('------- Covariance matrix of brief features. -------')
    print(cov_mat)


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


def visualize_brief_experiment():
    brief_coords = load_coords()
    train_images, train_labels, test_images, test_labels = load_actual_mnist()
    idxs = [0, 14, 45, 908, 12, 17]
    img_with_briefs = [cv2.resize(plot_brief_features_on_image(image, brief_coords), (200, 200)) for image in
                       train_images[idxs]]
    show_images(img_with_briefs)


def _main():
    # dataset = load_dataset()
    # images = dataset.get_attr('image')
    # labels = dataset.get_attr('label')
    # brief_features = extract_brief(images)

    # pruned_features = select_with_variance_threshold(brief_features, p=0.65)
    # k_best_features = select_with_k_best(brief_features, labels, num_features=60)

    # plot_ref = run_pca_experiments(image=images, labels=labels, brief_features=brief_features,
    #                                pruned_features=pruned_features,
    #                                k_best_features=k_best_features)
    #
    # plot_ref = run_mds_experiments(image=images, labels=labels, brief_features=brief_features,
    #                                pruned_features=pruned_features,
    #                                k_best_features=k_best_features)
    #
    # plot_ref.show()

    # run_covariance_inspect_experiment(brief_features)

    # run_knn_experiment_set([1, 2, 3, 4, 5], ['uniform', 'distance'], [16, 32, 64], images, labels)
    # run_knn_on_full_mnist_experiment([4])
    # run_knn_on_full_mnist_avg_images_brief_experiment(n_neighbors=4, desc_size=32)
    # run_decision_tree_with_brief_experiment()
    # run_svm_with_brief_experiment()
    visualize_brief_experiment()
    # run_svm_with_raw_images_experiment()

    # run_image_averaging_experiment(images, labels)

    # loaded_images = dataset.get_attr('image')
    # show_images(loaded_images)


if __name__ == '__main__':
    seed(SEED)
    _main()
