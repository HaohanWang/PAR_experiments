
import os

from datetime import datetime
import numpy as np
import imutils
import cv2
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
borderType = cv2.BORDER_CONSTANT
start = datetime.now()
batch_size = 128
num_classes = 10
epochs = 21
def check(img,y):
    print(y)
    cv2.imshow('1',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
import cPickle as pickle
import gzip
# f = gzip.open('../../../data/mnist.pkl.gz', 'rb')
# #f=gzip.open('/media/student/Data/zexue/MNIST/mnist.pkl.gz','rb')
# training_data, validation_data, test_data = pickle.load(f)
# x_train=training_data[0]
# y_train=training_data[1]
# x_val=validation_data[0]
# y_val=validation_data[1]
# x_test=test_data[0]
# y_test=test_data[1]
# x_train=x_train.reshape(x_train.shape[0],28,28)
# x_test=x_test.reshape(x_test.shape[0],28,28)
# x_val=x_val.reshape(x_val.shape[0],28,28)
# print (x_train.shape)
# print(x_train.astype)
# print(x_val.astype)

# def rotate(x,y):
#     rotatedx = []
#     rotatedy = []
#     rotatedp=[]
#     angles=[0,15,45,60,75]
#     for dig, lab in zip(x,y):
#         i=np.random.randint(5)
#         rotated = imutils.rotate(dig, angles[i])
#         rotatedx.append(rotated.reshape(784))
#         rotatedy.append(lab)
#         rotatedp.append(i)
#     rox=np.array(rotatedx)
#     roy=np.array(rotatedy)
#     rop=np.array(rotatedp)
#     return rox,roy,rop

# xtrain,ytrain,ptrain=rotate(x_train,y_train)
# print xtrain.shape
#check(x_train[19],ytrain[19])

#check(xtrain[19].reshape(28,28),ptrain[19])

# xval,yval,pval=rotate(x_val,y_val)
# xtest,ytest,ptest=rotate(x_test,y_test)
# np.save('../../../data/MNIST-r/npy/xtrain.npy',xtrain)
# np.save('../../../data/MNIST-r/npy/ytrain.npy',ytrain)
# np.save('../../../data/MNIST-r/npy/ptrain.npy',ptrain)
#
# np.save('../../../data/MNIST-r/npy/xval.npy',xval)
# np.save('../../../data/MNIST-r/npy/yval.npy',yval)
# np.save('../../../data/MNIST-r/npy/pval.npy',pval)
#
# np.save('../../../data/MNIST-r/npy/xtest.npy',xtest)
# np.save('../../../data/MNIST-r/npy/ytest.npy',ytest)
# np.save('../../../data/MNIST-r/npy/ptest.npy',ptest)

def oneHotRepresentation(y):
    n = y.shape[0]
    r = np.zeros([n, 10])
    for i in range(r.shape[0]):
        r[i,int(y[i])] = 1
    return r

def rotateImg_TrainVal(x, angles):
    rotatedx = []
    rotatedp=[]
    for dig in x:
        i=np.random.randint(5)
        rotated = imutils.rotate(dig, angles[i])
        rotatedx.append(rotated.reshape(784))
        rotatedp.append(i)
    rox=np.array(rotatedx)
    rop=np.array(rotatedp)
    return rox,rop

def rotateImg_Test(x, angle):
    rotatedx = []
    for dig in x:
        rotated = imutils.rotate(dig, angle)
        rotatedx.append(rotated.reshape(784))
    rox=np.array(rotatedx)
    return rox

def rotateImg(x, angle):
    x = x.reshape(x.shape[0],28,28)
    rotatedx = []
    for dig in x:
        rotated = imutils.rotate(dig, angle)
        rotatedx.append(rotated.reshape(28*28))
    rox=np.array(rotatedx)
    return rox

def subSamplingImages(X, y):
    indices = np.random.permutation(X.shape[0])
    X = X[indices,:]
    y = y[indices]
    mx = []
    my = []
    count = [0]*10
    for i in range(X.shape[0]):
        if count[y[i]] < 100:
            count[y[i]] += 1
            mx.append(X[i,:])
            my.append(y[i])
    return np.array(mx), np.array(my)

def loadDataRotate(test=0):
    # np.random.seed(0)

    if test == 0:
        trainAng = [15, 30, 45, 60, 75]
        testAng = 0
    elif test == 1:
        trainAng = [0, 30, 45, 60, 75]
        testAng = 15
    elif test == 2:
        trainAng = [0, 15, 45, 60, 75]
        testAng = 30
    elif test == 3:
        trainAng = [0, 15, 30, 60, 75]
        testAng = 45
    elif test == 4:
        trainAng = [0, 15, 30, 45, 75]
        testAng = 60
    else:
        trainAng = [0, 15, 30, 45, 60]
        testAng = 75

    f = gzip.open('../data/MNIST/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f)

    # x_train=training_data[0]
    # y_train=training_data[1]
    # x_val=validation_data[0]
    # y_val=validation_data[1]
    # x_test=test_data[0]
    # y_test=test_data[1]
    # x_train=x_train.reshape(x_train.shape[0],28,28)
    # x_test=x_test.reshape(x_test.shape[0],28,28)
    # x_val=x_val.reshape(x_val.shape[0],28,28)
    #
    #
    #
    # x_train, r_train = rotateImg_TrainVal(x_train, trainAng)
    # x_val, r_val = rotateImg_TrainVal(x_val, trainAng)
    # x_test = rotateImg_Test(x_test, testAng)
    #
    # return x_train, oneHotRepresentation(y_train),x_val,oneHotRepresentation(y_val),x_test,oneHotRepresentation(y_test)

    mx, my = subSamplingImages(training_data[0], training_data[1])
    trainValX = None
    trainValy = None
    for i in trainAng:
        if trainValX is None:
            trainValX = rotateImg(mx, i)
            trainValy = my
        else:
            trainValX = np.append(trainValX, rotateImg(mx, i), 0)
            trainValy = np.append(trainValy, my, 0)

    n = trainValX.shape[0]
    indices = np.random.permutation(n)
    trainX = trainValX[indices[:int(0.8*n)], :]
    trainY = trainValy[indices[:int(0.8*n)]]
    valX = trainValX[indices[int(0.8*n):], :]
    valY = trainValy[indices[int(0.8*n):]]

    testX = rotateImg(mx, testAng)
    testY = my

    return trainX, oneHotRepresentation(trainY), valX, oneHotRepresentation(valY), testX, oneHotRepresentation(testY)

