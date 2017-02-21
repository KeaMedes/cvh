from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from datetime import datetime
import numpy as np

from cifar import rgb_to_gray


def run_with_time(msg, func):
    begin = datetime.now()
    ret = func()
    end = datetime.now()
    dur = end - begin
    print("%s finish, time usage: %f" % (msg, dur.total_seconds()))
    return ret


def to_gray(train_data_color, test_data_color):
    train_data = np.array([rgb_to_gray(x) for x in train_data_color])
    test_data = np.array([rgb_to_gray(x) for x in test_data_color])
    return train_data, test_data


def knn_accuracy(train_data, train_label, test_data, test_label, k):
    """Use knn to classify and return the accuracy"""
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(train_data, train_label)
    result = knn.predict(test_data)

    accuracy = accuracy_score(y_true=test_label, y_pred=result)
    return accuracy


def linear_classifier_accuracy(train_data, train_label, test_data, test_label):
    clf = linear_model.SGDClassifier(loss='log')
    clf.fit(train_data, train_label)
    result = clf.predict(test_data)

    accuracy = accuracy_score(y_true=test_label, y_pred=result)
    return accuracy
