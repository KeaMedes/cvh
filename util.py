from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
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

    train_data = train_data.reshape(train_data.shape[0], -1)
    test_data = test_data.reshape(test_data.shape[0], -1)
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(train_data, train_label)
    result = knn.predict(test_data)

    accuracy = accuracy_score(y_true=test_label, y_pred=result)
    return accuracy
