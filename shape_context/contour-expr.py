from common.dataset_utils import load_actual_mnist
from common.img_utils import show_images

if __name__ == '__main__':
    train_images, train_labels, _, _ = load_actual_mnist()
    images_of_6 = train_images[train_labels == 6][0:1]
    show_images(images_of_6)
