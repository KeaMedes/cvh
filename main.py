from cifar import CiFar10
from classification import knn_accuracy
from preprocess import to_gray, dimension_reduction


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
        train_data, test_data = dimension_reduction(train_data, test_data, method=dim_reduce)
        print('transformed train data size: %dx%d' % (train_data.shape))
        print('transformed test data size: %dx%d' % (test_data.shape))

    accuracy = knn_accuracy(train_data, train_label, test_data, test_label)
    print("predict accuray: %f" % accuracy)


if __name__ == '__main__':
    main(gray=True, dim_reduce=None)
