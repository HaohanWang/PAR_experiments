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


def loadDataCifar10():
    Xtrain = np.load('../data/cifar10/trainData.npy')
    Ytrain = np.load('../data/cifar10/trainLabel.npy').astype(int)
    Xval = np.load('../data/cifar10/valData.npy')
    Yval = np.load('../data/cifar10/valLabel.npy').astype(int)
    Xtest = np.load('../data/cifar10/testData.npy')
    Ytest = np.load('../data/cifar10/testLabel.npy').astype(int)

    return Xtrain, oneHotRepresentation(Ytrain), Xval, oneHotRepresentation(Yval), Xtest, oneHotRepresentation(Ytest)
