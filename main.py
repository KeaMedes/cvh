from cifar import CiFar10, rgb_to_gray
from sklearn import random_projection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime
import numpy as np


def knn_accuracy(train_data, train_label, test_data, test_label):
    """Use knn to classify and return the accuracy"""
    knn = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)

    begin = datetime.now()
    knn.fit(train_data, train_label)
    end = datetime.now()
    dur = end - begin
    print("fit time: %f" % dur.total_seconds())

    begin = datetime.now()
    result = knn.predict(test_data)
    end = datetime.now()
    dur = end - begin
    print("predict time: %f" % dur.total_seconds())
    accuracy = accuracy_score(y_true=test_label, y_pred=result)
    return accuracy


def dim_red(train_data, test_data, method='rp'):
    """Do the dimension reduction based on the `method`"""
    train_size = train_data.shape[0]
    begin = datetime.now()
    all_data = np.concatenate([train_data, test_data])
    if method == 'rp':
        transformer = random_projection.SparseRandomProjection(eps=0.5)
        new_data = transformer.fit_transform(all_data)
    elif method == 'svd':
        pass
    end = datetime.now()
    dur = end - begin
    print("transform time: %f" % dur.total_seconds())
    return np.split(new_data, [train_size,])


def to_gray(train_data, test_data):
    begin = datetime.now()
    train_data = np.array([rgb_to_gray(x) for x in train_data])
    test_data = np.array([rgb_to_gray(x) for x in test_data])
    end = datetime.now()
    dur = end - begin
    print("transform to gray time: %f" % dur.total_seconds())
    return train_data, test_data


def main(gray=False, dim_reduce=None):
    cifar = CiFar10()
    cifar.load_data()

    train_data = cifar.get_train_data()
    train_label = cifar.get_train_label()
    test_data = cifar.get_test_data()
    test_label = cifar.get_test_label()

    if gray:
        train_data, test_data = to_gray(train_data, test_data)

    print('train data size: %dx%d' % (train_data.shape))
    print('test data size: %dx%d' % (test_data.shape))

    if dim_reduce:
        train_data, test_data = dim_red(train_data, test_data, method=dim_reduce)
        print('transformed train data size: %dx%d' % (train_data.shape))
        print('transformed test data size: %dx%d' % (test_data.shape))

    accuracy = knn_accuracy(train_data, train_label, test_data, test_label)
    print("predict accuray: %f" % accuracy)

if __name__ == '__main__':
    main(gray=True, dim_reduce=None)