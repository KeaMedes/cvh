import os
import cPickle


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


if __name__ == '__main__':
    cifar10 = CiFar10()
    cifar10.load_data('test_batch')