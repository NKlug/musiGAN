import numpy as np

from bar import to_one_hot


def create_major_scale(offset):
    return offset + np.asarray([0, 2, 4, 5, 7, 9, 11, 12])


def create_minor_scale(offset):
    return offset + np.asarray([0, 2, 3, 5, 7, 8, 10, 12])


def create_train_dataset():
    # only training with c, e, g, a

    offsets = [0, 1, 2, 3, 5, 6, 8, 10, 4, 7, 9]
    train_data = []
    for i in range(4):
        for offset in offsets:
            scale = create_major_scale(offset + 12 * i)
            rev = scale[::-1]
            one_hot = to_one_hot(scale)
            one_hot_rev = to_one_hot(rev)
            train_data.append(one_hot)
            # train_data.append(one_hot_rev)
    return np.random.permutation(np.asarray(train_data))
