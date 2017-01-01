from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from util import run_with_time


def knn_accuracy(train_data, train_label, test_data, test_label):
    """Use knn to classify and return the accuracy"""
    def knn_train():
        knn = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
        knn.fit(train_data, train_label)
        return knn
    knn = run_with_time('knn training', knn_train)

    def knn_predict():
        result = knn.predict(test_data)
        return result
    result = run_with_time('knn predict', knn_predict)

    accuracy = accuracy_score(y_true=test_label, y_pred=result)
    return accuracy
