import random

from cifar import CiFar10
from util import to_gray
from sklearn.cluster import KMeans
from sklearn.decomposition import SparseCoder
from sklearn.metrics import silhouette_score
import numpy as np
import cv2


def main(classify_method, k):
    cifar = CiFar10()
    cifar.load_data()

    print("loading data")
    train_data = cifar.get_train_data()
    train_label = cifar.get_train_label()
    test_data = cifar.get_test_data()
    test_label = cifar.get_test_label()
    train_data, test_data = to_gray(train_data, test_data)
    print("loading data done")

    # get the sift set of all images
    print("generating dense sift")
    sift_set = []
    for i in random.sample(range(0, 10000), 100):
        img = train_data[i].astype(np.uint8).reshape(32, 32)
        dense = cv2.FeatureDetector_create("Dense")
        sift = cv2.SIFT()
        kp = dense.detect(img)
        kp, des = sift.compute(img, kp)
        sift_set.append(des)
    sift_set = np.concatenate(sift_set)
    print sift_set.shape
    print("generating dense sift done")

    print("kmeans clustering")
    # cluster the sift-descriptors with k-means
    for cluster_num in range(30, 70, 5):
        kmeans = KMeans(n_clusters=cluster_num, random_state=10).fit(sift_set)
        cluster_labels = kmeans.predict(sift_set)
        silhouette_avg = silhouette_score(sift_set, cluster_labels)
        print("for n_cluster=%d, avg: %f" % (cluster_num, silhouette_avg))
    print("kmeans clustering done")

    # print("sparse coding")
    # coder = SparseCoder(dictionary=kmeans.cluster_centers_).fit(sift_set)
    # sift_set_sp = coder.transform(sift_set)
    # print sift_set_sp.shape
    # print("sparse coding done")


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
