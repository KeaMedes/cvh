import numpy as np
from cifar import CiFar10
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from itertools import product
from util import run_with_time, to_gray, knn_accuracy, linear_classifier_accuracy


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


def main(color, dr_method, dr_N_list, clf_method, KNN_K_list, nor):
    cifar = CiFar10()
    cifar.load_data()

    train_data = cifar.get_train_data()
    train_label = cifar.get_train_label()
    test_data = cifar.get_test_data()
    test_label = cifar.get_test_label()

    if color == 'gray':
        train_data, test_data = to_gray(train_data, test_data)

    print("data loading done, color: %s" % color)

    # data normalization
    if nor:
        scaler = StandardScaler().fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)

    print("data normalization done, do normalization: %d" % nor)

    for dr_N in dr_N_list:
        # dimension reduction
        if dr_method == 'rp':
            transformer = SparseRandomProjection(n_components=dr_N).fit(train_data)
        else:
            transformer = PCA(n_components=dr_N).fit(train_data)
        new_train_data = transformer.transform(train_data)
        new_test_data = transformer.transform(test_data)
        print("dimension reduction done, method: %s, N: %d" % (dr_method, dr_N))

        # get the accuracy
        if clf_method == 'knn':
            for KNN_k in KNN_K_list:
                accuracy = knn_accuracy(new_train_data, train_label, new_test_data, test_label, KNN_k)
                print ("with KNN, and K: %d, get accuracy: %f" % (KNN_k, accuracy))
        else:
            accuracy = linear_classifier_accuracy(new_train_data, train_label, new_test_data, test_label)
            print("with linear classfier, get accuracy: %f" % accuracy)
    print("-----------------------------------------------------------")


if __name__ == '__main__':
    reduced_dim_list = [200, 300, 500]
    KNN_k_list = [1, 5, 9]
    main(color='color', dr_method='rp', dr_N_list=reduced_dim_list, clf_method='linear', KNN_K_list=None, nor=True)
    main(color='color', dr_method='svd', dr_N_list=reduced_dim_list, clf_method='linear', KNN_K_list=None, nor=True)
    main(color='color', dr_method='rp', dr_N_list=reduced_dim_list, clf_method='knn', KNN_K_list=KNN_k_list, nor=True)
    main(color='color', dr_method='svd', dr_N_list=reduced_dim_list, clf_method='knn', KNN_K_list=KNN_k_list, nor=True)
