import random

from cifar import CiFar10
from util import to_gray, knn_accuracy
from sklearn.cluster import KMeans
from sklearn.decomposition import SparseCoder
from sklearn.metrics import silhouette_score
import numpy as np
import cv2


def generate_dense_sift(X):
    dense = cv2.FeatureDetector_create("Dense")
    sift = cv2.SIFT()
    sift_set = []
    for img in X:
        img = img.astype(np.uint8).reshape(32, 32)
        kp = dense.detect(img)
        kp, des = sift.compute(img, kp)
        sift_set.append(des)
    sift_set = np.concatenate(sift_set)
    return sift_set.reshape(X.shape[0], -1, 128)


def random_pick(X, n):
    sift_set = []
    for i in random.sample(xrange(0, X.shape[0]), n):
        tmp_sift_set = X[i]
        sift_set.append(tmp_sift_set)
    return np.concatenate(sift_set)


def kmeas(X_kmeans, k):
    kmeans = KMeans(n_clusters=k, random_state=10).fit(X_kmeans)
    cluster_labels = kmeans.predict(X_kmeans)
    silhouette_avg = silhouette_score(X_kmeans, cluster_labels)
    print("for n_cluster=%d, avg: %f" % (k, silhouette_avg))
    return kmeans.cluster_centers_


def sparse_coding(X, X_base):
    new_X = X.reshape(-1, 128)
    coder = SparseCoder(dictionary=X_base).fit(new_X)
    return coder.transform(new_X).reshape(X.shape[0], X.shape[1], -1)


def sp_average_pooling(X):
    new_X = []
    for img in X:
        patch_list = []
        for i in range(0, 6, 2):
            for j in range(0, 6, 2):
                s0 = i * 6 + j
                s1 = s0 + 1
                s2 = s0 + 6
                s3 = s2 + 1
                patch = img[[s0, s1, s2, s3]]
                patch = np.mean(patch, axis=0)
                patch_list.append(patch)
        level1 = np.stack(patch_list)
        level2 = np.mean(level1, axis=0)
        patch_list.append(level2)
        sp = np.stack(patch_list)
        new_X.append(sp)
    return np.stack(new_X)


def main(classify_method, k):
    cifar = CiFar10()
    cifar.load_data()

    print("loading data")
    X_train = cifar.get_train_data()
    Y_train = cifar.get_train_label()
    X_test = cifar.get_test_data()
    Y_test = cifar.get_test_label()
    X_train, X_test = to_gray(X_train, X_test)
    print("loading data done, X_train shape: %s, X_test shape: %s" % (str(X_train.shape), str(X_test.shape)))

    # generate dense sift for all images
    print("generating dense sift")
    X_train = generate_dense_sift(X_train)
    X_test = generate_dense_sift(X_test)
    print("generating dense sift done, X_train shape: %s, X_test shape: %s" % (str(X_train.shape), str(X_test.shape)))

    # random pick 200 images
    print("random pick 200 images")
    X_kmeans = random_pick(X_train, 5000)
    print("random pick 200 images done, X_kmeans shape: %s" % (str(X_kmeans.shape)))

    # cluster the sift-descriptors with k-means
    print("kmeans clustering")
    X_base = kmeas(X_kmeans, 16)
    print("kmeans clustering done, X_base shape: %s" % str(X_base.shape))

    # sparse coding
    print("spare coding")
    X_train = sparse_coding(X_train, X_base)
    X_test = sparse_coding(X_test, X_base)
    print("spare coding done, X_train shape: %s, X_test shape: %s" % (str(X_train.shape), str(X_test.shape)))

    # spatial pyramid average pooling
    X_train = sp_average_pooling(X_train)
    X_test = sp_average_pooling(X_test)
    print("sp average pooling done, X_train shape: %s, X_test shape: %s" % (str(X_train.shape), str(X_test.shape)))

    # get accuracy
    accuracy = knn_accuracy(X_train, Y_train, X_test, Y_test, 5)
    print accuracy


def test_sift():
    img = 'test.jpg'
    img = cv2.imread(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print img_gray.shape
    dense = cv2.FeatureDetector_create("Dense")
    sift = cv2.SIFT()
    kp = dense.detect(img_gray)
    kp, des = sift.compute(img_gray, kp)
    print des.shape
    print des


if __name__ == '__main__':
    main('', '')
    # test_sift()
