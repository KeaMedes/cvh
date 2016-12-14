import os
import cPickle
import matplotlib.pyplot as plt
import numpy as np


def rgb_to_gray(img):
    # img = img_r.astype(np.float32) * 0.2989 + img_b.astype(np.float32) * 0.1140 + img_g.astype(np.float32) * 0.5870
    # img = img_r * 0.2989 + img_b * 0.1140 + img_g.astype * 0.5870
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])


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

    def get_test_img(self):
        return self.raw_data[self.data_batchs[0]]['data'][0]

    def show_test_img(self, img):
        img = img.reshape(3, 32, 32)
        plt.subplot(2, 2, 1)
        plt.imshow(img[0], cmap='gray')
        plt.subplot(2, 2, 2)
        plt.imshow(img[1], cmap='gray')
        plt.subplot(2, 2, 3)
        plt.imshow(img[2], cmap='gray')
        plt.subplot(2, 2, 4)
        img_gray = rgb_to_gray(img.reshape(1024 * 3))
        plt.imshow(img_gray, cmap='gray')
        plt.show()


if __name__ == '__main__':
    cifar10 = CiFar10()
    cifar10.load_data()
    img = cifar10.get_test_img()
    cifar10.show_test_img(img)