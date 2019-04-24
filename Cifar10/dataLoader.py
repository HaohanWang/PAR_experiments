import numpy as np

def oneHotRepresentation(y, num=10):
    r = []
    for i in range(y.shape[0]):
        l = np.zeros(num)
        l[y[i]] = 1
        r.append(l)
    return np.array(r)

def loadDataCifar10():
    Xtrain = np.load('../../data/cifar10/trainData.npy').astype(np.float)
    Ytrain = np.load('../../data/cifar10/trainLabel.npy').astype(int)
    Xval = np.load('../../data/cifar10/valData.npy').astype(np.float)
    Yval = np.load('../../data/cifar10/valLabel.npy').astype(int)
    Xtest = np.load('../../data/cifar10/testData.npy').astype(np.float)
    Ytest = np.load('../../data/cifar10/testLabel.npy').astype(int)

    return Xtrain, oneHotRepresentation(Ytrain), Xval, oneHotRepresentation(Yval), Xtest, oneHotRepresentation(Ytest)

def loadDataCifar10_2():
    Xtrain = np.load('../../data/cifar10/trainData2.npy').astype(np.float)
    Ytrain = np.load('../../data/cifar10/trainLabel2.npy').astype(int)
    Xtest = np.load('../../data/cifar10/testData.npy').astype(np.float)
    Ytest = np.load('../../data/cifar10/testLabel.npy').astype(int)

    # padding for data augmentation
    return Xtrain, oneHotRepresentation(Ytrain), Xtest, oneHotRepresentation(Ytest)


def loadCifarTest():
    Xtest_g = np.load('../../data/cifar10/testData_greyscale.npy').astype(np.float)
    Xtest_n = np.load('../../data/cifar10/testData_negative.npy').astype(np.float)
    Ytest = np.load('../../data/cifar10/testLabel.npy').astype(int)
