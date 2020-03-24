import numpy as np


def make_weights_for_balanced_classes(labels):
    assert isinstance(labels, np.ndarray)
    labels = labels[np.nonzero(labels)] - 1
    classes, counts = np.unique(labels, return_counts=True)
    nclasses = len(classes)
    # print("classes: ", classes)
    # print("counts: ", counts)
    weight_per_class = [0.] * nclasses
    N = np.float32(np.sum(counts))
    # print("N: ", N)
    for i in range(nclasses):
        weight_per_class[i] = N / np.float32(counts[i])
    weight = [0] * int(N)
    # print(weight)
    print(labels.shape)
    for idx, val in enumerate(labels):
        weight[idx] = weight_per_class[val]
    return weight