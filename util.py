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
