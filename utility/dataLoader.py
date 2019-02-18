__author__ = 'Haohan Wang'

import numpy as np
import cPickle


def oneHotRepresentation(y, num=10):
    r = []
    for i in range(y.shape[0]):
        l = np.zeros(num)
        l[y[i]] = 1
        r.append(l)
    return np.array(r)


def loadDataMNIST():
    f = open('../data/MNIST/mnist.pkl', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    return training_data[0], oneHotRepresentation(training_data[1]), validation_data[
        0], oneHotRepresentation(validation_data[1]), test_data[0], oneHotRepresentation(test_data[1])
