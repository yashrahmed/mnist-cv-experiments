import numpy as np
from sklearn.cluster import KMeans


# run k-means to generate exemplars
def generate_exemplar_dataset(images, labels, clusters_per_label=20):
    uniq_labels = np.unique(labels)
    _, h, w = images.shape
    exemplars = None
    k_means = KMeans(n_clusters=clusters_per_label)
    for label in uniq_labels:
        print(f'Generating exemplars for label = {label}')
        img_idxs = np.where(labels == label)
        images_for_label = images[img_idxs].reshape([-1, h * h])
        k_means.fit(images_for_label)
        centers = k_means.cluster_centers_
        for i in range(centers.shape[0]):
            centers[i] = np.interp(centers[i], [np.min(centers[i]), np.max(centers[i])], [0, 255])
        exemplars = np.concatenate((exemplars, centers), axis=0) if exemplars is not None else centers
    return exemplars.astype(np.uint8).reshape([-1, h, h]), np.repeat(uniq_labels, clusters_per_label)
