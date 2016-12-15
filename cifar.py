import os
import cPickle
import matplotlib.pyplot as plt
import numpy as np


def rgb_to_gray(img):
    return np.dot([0.299, 0.587, 0.114], img.reshape(3, 1024))

def to_read_img_slow(img):
    new_img = []
    img = img.reshape(3, 1024)
    for i in range(0, 1024):
        new_img.append(img[0][i])
        new_img.append(img[1][i])
        new_img.append(img[2][i])
    return np.array(new_img)


class CiFar10(object):
    def __init__(self, dir='cifar10'):
        self.dir = dir
        self.data_batchs = []
        self.test_batchs = []
        self.raw_data = {}
        for file in os.listdir(self.dir):
            if file.startswith('data_batch'):
                self.data_batchs.append(file)
            elif file.startswith('test_batch'):
                self.test_batchs.append(file)

    def _load_batch(self, target):
        with open(os.path.join(self.dir, target), 'rb') as fin:
            dict = cPickle.load(fin)
            self.raw_data[target] = dict

    def load_data(self, target=None):
        if target:
            if target in self.data_batchs or target in self.test_batchs:
                self._load_batch(target)
        else:
            for target in self.data_batchs + self.test_batchs:
                self._load_batch(target)

    def get_test_img(self, index=0):
        return self.raw_data[self.data_batchs[0]]['data'][index]

    def show_test_img(self, img):
        img = img.reshape(3, 32, 32)
        plt.subplot(2, 3, 1)
        plt.imshow(img[0], cmap='gray')
        plt.subplot(2, 3, 2)
        plt.imshow(img[1], cmap='gray')
        plt.subplot(2, 3, 3)
        plt.imshow(img[2], cmap='gray')
        plt.subplot(2, 3, 4)
        img_gray = rgb_to_gray(img).reshape(32, 32)
        plt.imshow(img_gray, cmap='gray')
        plt.subplot(2, 3, 5)
        img_color = to_read_img_slow(img)
        plt.imshow(img_color.reshape(32, 32, 3))
        plt.show()

    def get_data(self, name=None):
        if name in self.data_batchs or name in self.test_batchs:
            return self.raw_data[name]['data']
        else:
            all = []
            for name in self.data_batchs:
                all.append(self.raw_data[name]['data'])
            return np.concatenate(all)

    def get_label(self, name=None):
        if name in self.data_batchs or name in self.test_batchs:
            return np.array(self.raw_data[name]['labels'])
        else:
            all = []
            for name in self.data_batchs:
                all.append(self.raw_data[name]['labels'])
            return np.concatenate(all)

    def get_train_data(self):
        return self.get_data()

    def get_train_label(self):
        return self.get_label()

    def get_test_data(self):
        return self.get_data(self.test_batchs[0])

    def get_test_label(self):
        return self.get_label(self.test_batchs[0])


if __name__ == '__main__':
    cifar10 = CiFar10()
    cifar10.load_data()
    # img = cifar10.get_test_img(101)
    # cifar10.show_test_img(img)
    # data = cifar10.get_data()
    # transformer = random_projection.SparseRandomProjection(eps=0.9)
    # new_data = transformer.fit_transform(data)
    # print data.shape
    # print new_data.shape
    test_data = cifar10.get_test_data()
    test_label = cifar10.get_test_label()
    print test_data.shape
    print test_label.shape
