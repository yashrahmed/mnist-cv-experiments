import matplotlib.pyplot as plt


def scatter_plot(dataset, labels, title='Default Title'):
    plt.figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']
    targets = list(range(0, 10))  # possible category labels
    for color, target in zip(colors, targets):
        idxs = labels == target
        plt.scatter(dataset[idxs, 0], dataset[idxs, 1], color=color, alpha=0.8, label=target)
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title(title)
    return plt
