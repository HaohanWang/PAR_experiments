# -*- encoding=utf-8 -*-
import cv2
# Standard library
import time
import gzip
import zipfile
import numpy as np
import os
import math
from PIL import Image  
import matplotlib.pyplot as plt
import cPickle as pickle
import glob

np.random.seed(0)

radial=[i*0.1*14 for i in range(10)]
random = [i*0.1 for i in range(10)]

def fft(img):
    return np.fft.fft2(img)
def fftshift(img):
    return np.fft.fftshift(fft(img))
def ifft(img):
    return np.fft.ifft2(img)
def ifftshift(img):
    return ifft(np.fft.ifftshift(img))

def oneHotRepresentation(y):
    n = y.shape[0]
    r = np.zeros([n, 10])
    for i in range(r.shape[0]):
        r[i,int(y[i])] = 1
    return r

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


def mask_radial(img,l,isGray=True): 
    rows,cols=img.shape
    mask = np.zeros((rows,cols)) 
    for i in range(rows):
        for j in range(cols):
            mask[i,j]=distance2(i,j,rows,cols,r=14, l=l)
    return mask

def mask_random(img,p = 0.5,isGray=True):  
    rows,cols=img.shape
    mask=np.random.binomial(1,p,(rows,cols))
    return mask

maskRandomKernel = [mask_random(np.zeros([28,28]),random[i],True) for i in range(10)]
maskRadioKernel = [mask_radial(np.zeros([28,28]),i,True) for i in range(10)]

def change_img_random(image,r,g,b,isGray=True,label=-1):
    if label==-1:
        x=np.random.randint(10)
    else:
        # label=0-9
        if np.random.random() < 0.8:
            x=label
        else:
            x = np.random.randint(10)

    _mask = maskRandomKernel[x]

    if isGray == True:
        fftshift_img_r=fftshift(r)
        fftshift_result_r = fftshift_img_r * _mask
        result_r = ifftshift(fftshift_result_r) 
        mr=np.abs(result_r)
        return mr

    fftshift_img_b=fftshift(b)
    fftshift_result_b = fftshift_img_b * _mask
    result_b = ifftshift(fftshift_result_b) 
    mb=np.abs(result_b)
    
    _mask=mask_random(image,0.0)
    fftshift_img_g=fftshift(g)
    fftshift_result_g = fftshift_img_g * _mask
    result_g = ifftshift(fftshift_result_g) 
    mg=np.abs(result_g)

    _mask=mask_random(image,0.0)
    fftshift_img_r=fftshift(r)
    fftshift_result_r = fftshift_img_r * _mask
    result_r = ifftshift(fftshift_result_r) 
    mr=np.abs(result_r)

    img_mix=cv2.merge([mr,mg,mb])
    return img_mix

def change_img_radial(image,r,g,b,isGray=True,label=-1, corr=0.8):
    if label==-1:
        x=np.random.randint(10)
    else:
        if np.random.random() < corr:
            x=label
        else:
            x = np.random.randint(10)
            
    _mask = maskRadioKernel[x]

    if isGray == True:
        fftshift_img_r=fftshift(r)
        fftshift_result_r = fftshift_img_r * _mask
        result_r = ifftshift(fftshift_result_r) 
        mr=np.abs(result_r)
        return mr, x
        
    

    fftshift_img_b=fftshift(b)
    fftshift_result_b = fftshift_img_b * _mask
    result_b = ifftshift(fftshift_result_b) 
    mb=np.abs(result_b)

    fftshift_img_g=fftshift(g)
    fftshift_result_g = fftshift_img_g * _mask
    result_g = ifftshift(fftshift_result_g) 
    mg=np.abs(result_g)

    fftshift_img_r=fftshift(r)
    fftshift_result_r = fftshift_img_r * _mask
    result_r = ifftshift(fftshift_result_r) 
    mr=np.abs(result_r)

    img_mix=cv2.merge([mr,mg,mb])
    return img_mix, x

def one_hot(x, n):
    """
    convert index representation to one-hot representation
    """
    x = np.array(x)
    assert x.ndim == 1
    return np.eye(n)[x]


def _grayscale(a):
    return a.reshape(a.shape[0], 3, 32, 32).mean(1).reshape(a.shape[0], -1)


def change_mnist_radial(isGray=True, corr=0.8):
    f = gzip.open('../data/MNIST/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f)
    f.close()

    _Xtrain=np.zeros((training_data[0].shape[0],28*28))
    _Xvalidation = np.zeros((validation_data[0].shape[0],28*28))
    _Xtest=np.zeros((test_data[0].shape[0],28*28))
    idLabel_train = []
    idLabel_val = []
    idLabel_test = []
    for i in range(training_data[0].shape[0]):
        r = training_data[0][i]
        r=r.reshape(28,28)
        img_mix, pid =change_img_radial(r,r,r,r,isGray,int(training_data[1][i]),corr=corr)
        _Xtrain[i]=img_mix.reshape(1,28*28)
        idLabel_train.append(pid)

    for i in range(validation_data[0].shape[0]):
        r = validation_data[0][i]
        r=r.reshape(28,28)
        img_mix, pid=change_img_radial(r,r,r,r,isGray,int(validation_data[1][i]),corr=corr)
        _Xvalidation[i]=img_mix.reshape(1,28*28)
        idLabel_val.append(pid)

    for i in range(test_data[0].shape[0]):
        r = test_data[0][i]
        r=r.reshape(28,28)
        img_mix, pid=change_img_radial(r,r,r,r,isGray,-1,corr=corr)
        _Xtest[i]=img_mix.reshape(1,28*28)
        idLabel_test.append(pid)

    idLabel_train = np.array(idLabel_train)
    idLabel_val = np.array(idLabel_val)
    idLabel_test = np.array(idLabel_test)

    #shuffle

    training_label = training_data[1]

    indices = np.random.permutation(_Xtrain.shape[0])
    _Xtrain = _Xtrain[indices, :]
    training_label = training_data[1][indices]

    return _Xtrain, oneHotRepresentation(training_label),_Xvalidation,oneHotRepresentation(validation_data[1]),_Xtest,oneHotRepresentation(test_data[1])

####################################################################################
### MultiDomain Codes 
####################################################################################

def addingPattern(r, mask):
    fftshift_img_r=fftshift(r)
    fftshift_result_r = fftshift_img_r * mask
    result_r = ifftshift(fftshift_result_r)
    mr=np.abs(result_r)
    return mr

def mask_radial_MM(isGray=True):  
    mask = np.zeros((28,28)) 
    for i in range(28):
        for j in range(28):
            mask[i,j]=distance(i,j,28,28,r=3.5)
    return mask

def mask_random_MM(p = 0.5,isGray=True):  
    mask=np.random.binomial(1,1-p,(28,28))
    return mask

def addMultiDomainPattern(r, l, testCase, testingFlag=False, randomMask=None, radioMask=None):
    if testingFlag:
        if testCase == 0:
            return r
        elif testCase == 1:
            return addingPattern(r, randomMask)
        else:
            return addingPattern(r, radioMask)
    else:
        # if l < 5:   # TODO: switch this line with the next line to control whether the experiment is on dependent case or independent case
        if np.random.random() < 0.5:
            k = 1
        else:
            k = 2
        return addMultiDomainPattern(r, None, int(testCase+k)%3, testingFlag=True, randomMask=randomMask, radioMask=radioMask)


def loadMultiDomainMNISTData(testCase=1):
    '''
    :param testCase:
            0 for original distribution as testing
            1 for random kernel as testing
            2 for radial kernel as testing
    :return:
    '''

    np.random.seed(1)

    f = gzip.open('../data/MNIST/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f)
    f.close()

    randomMask = mask_random_MM()
    radioMask = mask_radial_MM()

    _Xtrain=np.zeros((training_data[0].shape[0],28*28))
    _Xvalidation = np.zeros((validation_data[0].shape[0],28*28))
    _Xtest=np.zeros((test_data[0].shape[0],28*28))

    for i in range(training_data[0].shape[0]):
        r = training_data[0][i]
        r=r.reshape(28,28)
        img = addMultiDomainPattern(r, training_data[1][i], testCase, randomMask=randomMask, radioMask=radioMask)
        _Xtrain[i]=img.reshape(1,28*28)

    for i in range(validation_data[0].shape[0]):
        r = validation_data[0][i]
        r=r.reshape(28,28)
        img = addMultiDomainPattern(r, training_data[1][i], testCase, randomMask=randomMask, radioMask=radioMask)
        _Xvalidation[i]=img.reshape(1,28*28)


    for i in range(test_data[0].shape[0]):
        r = test_data[0][i]
        r=r.reshape(28,28)
        img = addMultiDomainPattern(r, training_data[1][i], testCase, testingFlag=True,randomMask=randomMask, radioMask=radioMask)
        _Xtest[i]=img.reshape(1,28*28)

    indices = np.random.permutation(_Xtrain.shape[0])
    _Xtrain = _Xtrain[indices, :]
    training_label = training_data[1][indices]

    return _Xtrain, oneHotRepresentation(training_label),_Xvalidation,oneHotRepresentation(validation_data[1]),_Xtest,oneHotRepresentation(test_data[1])

if __name__ == '__main__':
    loadMultiDomainMNISTData(testCase=2)

