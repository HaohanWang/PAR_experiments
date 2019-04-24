# -*- encoding=utf-8 -*-
# Standard library
import numpy as np

np.random.seed(0)

def fft(img):
    return np.fft.fft2(img)
def fftshift(img):
    return np.fft.fftshift(fft(img))
def ifft(img):
    return np.fft.ifft2(img)
def ifftshift(img):
    return ifft(np.fft.ifftshift(img))

def distance(i,j,w,h,r):
    dis=np.sqrt((i-14)**2+(j-14)**2)
    if dis<r:
        return 0.0
    else:
        return 1.0

def distance2(i,j,w,h,r=14.0,l=0):
    if l == 0:
        return 1
    elif l == 1:
        dis=np.sqrt((i-0)**2+(j-0)**2)
    elif l == 2:
        dis=np.sqrt((i-0)**2+(j-13.5)**2)
    elif l == 3:
        dis=np.sqrt((i-0)**2+(j-27)**2)
    elif l == 4:
        dis=np.sqrt((i-13.5)**2+(j-0)**2)
    elif l == 5:
        dis=np.sqrt((i-13.5)**2+(j-13.5)**2)
    elif l == 6:
        dis=np.sqrt((i-13.5)**2+(j-27)**2)
    elif l == 7:
        dis=np.sqrt((i-27)**2+(j-0)**2)
    elif l == 8:
        dis=np.sqrt((i-27)**2+(j-13.5)**2)
    else:
        dis=np.sqrt((i-27)**2+(j-27)**2)
    if dis < r:
        return 1
    else:
        return 0


def mask_radial_MM():
    mask = np.zeros((32,32))
    for i in range(32):
        for j in range(32):
            mask[i,j]=distance(i,j,32,32,r=3.5)
    return [mask, mask, mask]


def mask_random_MM(p = 0.5):
    mask_rgb = []
    for i in range(3):
        mask = np.random.binomial(1,1-p,(32,32))
        mask_rgb.append(mask)
    return mask_rgb


def oneHotRepresentation(y, num=10):
    r = []
    for i in range(y.shape[0]):
        l = np.zeros(num)
        l[y[i]] = 1
        r.append(l)
    return np.array(r)

def loadDataCifar10():
    Xtrain = np.load('../../data/cifar10/trainData2.npy').astype(np.float)
    Ytrain = np.load('../../data/cifar10/trainLabel2.npy').astype(int)
    # Xval = np.load('../../data/cifar10/valData.npy').astype(np.float)
    # Yval = np.load('../../data/cifar10/valLabel.npy').astype(int)
    Xtest = np.load('../../data/cifar10/testData.npy').astype(np.float)
    Ytest = np.load('../../data/cifar10/testLabel.npy').astype(int)
    return Xtrain, oneHotRepresentation(Ytrain), Xtest, oneHotRepresentation(Ytest)

def loadCifarTest():
    Xtest_g = np.load('../../data/cifar10/testData_greyscale.npy').astype(np.float)
    Xtest_n = np.load('../../data/cifar10/testData_negative.npy').astype(np.float)
    Ytest = np.load('../../data/cifar10/testLabel.npy').astype(int)


def addingPattern(r, mask):
    fftshift_img_r=fftshift(r[:, :, 0])
    fftshift_result_r = fftshift_img_r * mask[0]
    result_r = ifftshift(fftshift_result_r) 
    mr=np.abs(result_r).astype(int)
    fftshift_img_g=fftshift(r[:, :, 1])
    fftshift_result_g = fftshift_img_g * mask[1]
    result_g = ifftshift(fftshift_result_g) 
    mg=np.abs(result_g).astype(int)
    fftshift_img_b=fftshift(r[:, :, 2])
    fftshift_result_b = fftshift_img_b * mask[2]
    result_b = ifftshift(fftshift_result_b) 
    mb=np.abs(result_b).astype(int)
    return np.concatenate((mr[:, :, np.newaxis], mg[:, :, np.newaxis], mb[:, :, np.newaxis]), 2)


def addMultiDomainPattern(r, l, testCase, dependency=False, testingFlag=False, randomMask=None, radioMask=None):
    if testingFlag:
        if testCase == 0:
            return r
        elif testCase == 1:
            return addingPattern(r, randomMask)
        else:
            return addingPattern(r, radioMask)
    else:
        if dependency:
            if l < 5:
                k = 1
            else:
                k = 2
        else:
            if np.random.random() < 0.5:
                k = 1
            else:
                k = 2
        return addMultiDomainPattern(r, None, int(testCase+k)%3, None, testingFlag=True, randomMask=randomMask, radioMask=radioMask)


def loadMultiDomainCifar10Data(testCase=1, dependency=False):
    '''
    :param testCase:
            0 for original distribution as testing
            1 for random kernel as testing
            2 for radial kernel as testing
    :return:
    '''

    np.random.seed(1)

    Xtrain = np.load('../../data/cifar10/trainData2.npy').astype(np.float)
    Ytrain = np.load('../../data/cifar10/trainLabel2.npy').astype(int)
    # Xval = np.load('../../data/cifar10/valData.npy')
    # Yval = np.load('../../data/cifar10/valLabel.npy').astype(int)
    Xtest = np.load('../../data/cifar10/testData.npy').astype(np.float)
    Ytest = np.load('../../data/cifar10/testLabel.npy').astype(int)

    randomMask = mask_random_MM()
    radioMask = mask_radial_MM()

    _Xtrain=np.zeros((Xtrain.shape))
    _Xtest=np.zeros((Xtest.shape))

    for i in range(Xtrain.shape[0]):
        r = Xtrain[i]
        img = addMultiDomainPattern(r, Ytrain[i], testCase, dependency, randomMask=randomMask, radioMask=radioMask)
        _Xtrain[i]=img

    for i in range(Xtest.shape[0]):
        r = Xtest[i]
        img = addMultiDomainPattern(r, Ytest[i], testCase, dependency, testingFlag=True, randomMask=randomMask, radioMask=radioMask)
        _Xtest[i]=img

    indices = np.random.permutation(_Xtrain.shape[0])
    _Xtrain = _Xtrain[indices, :]
    Ytrain = Ytrain[indices]

    return _Xtrain, oneHotRepresentation(Ytrain), _Xtest, oneHotRepresentation(Ytest)
