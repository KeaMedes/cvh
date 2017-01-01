from sklearn import random_projection
import numpy as np

from cifar import rgb_to_gray
from util import run_with_time


def dimension_reduction(train_data, test_data, method='rp'):
    """Do the dimension reduction based on the `method`"""
    train_size = train_data.shape[0]
    new_data = None
    if method == 'rp':
        def dimension_reduction_rp():
            all_data = np.concatenate([train_data, test_data])
            transformer = random_projection.SparseRandomProjection(eps=0.5)
            new_data = transformer.fit_transform(all_data)
            return new_data
        new_data = run_with_time('dimension reduction with random projection', dimension_reduction_rp)
    elif method == 'svd':
        pass
    return np.split(new_data, [train_size,])


def to_gray(train_data_color, test_data_color):
    def to_gray_func():
        train_data = np.array([rgb_to_gray(x) for x in train_data_color])
        test_data = np.array([rgb_to_gray(x) for x in test_data_color])
        return train_data, test_data
    train_data, test_data = run_with_time('transform the image to gray scale', to_gray_func)
    return train_data, test_data
