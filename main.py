import numpy as np
from cifar import CiFar10
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from itertools import product


def knn_accuracy(train_data, train_label, test_data, test_label, k):
    """Use knn to classify and return the accuracy"""

    def knn_train():
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn.fit(train_data, train_label)
        return knn

    knn = run_with_time('knn training', knn_train)

    def knn_predict():
        result = knn.predict(test_data)
        return result

    result = run_with_time('knn predict', knn_predict)

    accuracy = accuracy_score(y_true=test_label, y_pred=result)
    return accuracy


from cifar import rgb_to_gray
from util import run_with_time


def dimension_reduction(train_data, test_data, method, N):
    """Do the dimension reduction based on the `method`"""
    train_size = train_data.shape[0]
    new_data = None
    if method == 'rp':
        def dimension_reduction_rp():
            all_data = np.concatenate([train_data, test_data])
            transformer = SparseRandomProjection(n_components=N)
            new_data = transformer.fit_transform(all_data)
            return new_data

        new_data = run_with_time('dimension reduction with random projection', dimension_reduction_rp)
    elif method == 'svd':
        def dimension_reduction_pca():
            all_data = np.concatenate([train_data, test_data])
            transformer = PCA(n_components=N)
            new_data = transformer.fit_transform(all_data)
            return new_data

        new_data = run_with_time('dimension reduction with svd', dimension_reduction_pca)
    return np.split(new_data, [train_size, ])


def to_gray(train_data_color, test_data_color):
    def to_gray_func():
        train_data = np.array([rgb_to_gray(x) for x in train_data_color])
        test_data = np.array([rgb_to_gray(x) for x in test_data_color])
        return train_data, test_data

    train_data, test_data = run_with_time('transform the image to gray scale', to_gray_func)
    return train_data, test_data


def main(image_color, dim_reduction_method, normalization_time, reduced_dim, KNN_K):
    cifar = CiFar10()
    cifar.load_data()

    train_data = cifar.get_train_data()
    train_label = cifar.get_train_label()
    test_data = cifar.get_test_data()
    test_label = cifar.get_test_label()

    if image_color == 'gray':
        train_data, test_data = to_gray(train_data, test_data)

    # data normalization
    scaler = StandardScaler().fit(train_data)
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.fit_transform(test_data)

    # dimension reduction
    if dim_reduction_method == 'rp':
        transformer = SparseRandomProjection(n_components=reduced_dim)
    else:
        transformer = PCA(n_components=reduced_dim)
    train_data = transformer.fit_transform(train_data)
    test_data = transformer.fit_transform(test_data)
    print train_data.shape, test_data.shape

    accuracy = knn_accuracy(train_data, train_label, test_data, test_label, KNN_K)
    return accuracy


if __name__ == '__main__':
    # color_list = ['color', 'gray']
    # dim_reduction_method_list = ['rp', 'svd']
    # normalization_time_list = ['before', 'after']
    # reduced_dim_list = [200, 300, 500]
    # KNN_k_list = [1, 5, 9]
    # for color, dim_reduction_method, normalization_time, reduced_dim, KNN_k in \
    #         product(color_list, dim_reduction_method_list, normalization_time_list, reduced_dim_list, KNN_k_list):
    #     accuracy = main(color, dim_reduction_method, normalization_time, reduced_dim, KNN_k)
    #     print('color: %s, method: %s, time: %s, N: %d, k: %d, accuracy: %f' % (color, dim_reduction_method,
    #                                                                            normalization_time, reduced_dim, KNN_k,
    #                                                                            accuracy))
    print main('gray', 'svd', 'before', 200, 9)
