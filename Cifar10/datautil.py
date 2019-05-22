import numpy as np

def horizontal_flip(image, axis):
    '''
    Flip an image at 50% possibility
    :param image: a 3 dimensional numpy array representing an image
    :param axis: 0 for vertical flip and 1 for horizontal flip
    :return: 3D image after flip
    '''
    flip_prop = np.random.randint(low=0, high=2)
    if flip_prop == 0:
        image = cv2.flip(image, axis)

    return image

def random_crop_and_flip(batch_data, padding_size):
    '''
    Helper to random crop and random flip a batch of images
    :param padding_size: int. how many layers of 0 padding was added to each side
    :param batch_data: a 4D batch array
    :return: randomly cropped and flipped image
    '''
    cropped_batch = np.zeros(len(batch_data) * 32 * 32 * 3).reshape(
        len(batch_data), 32, 32, 3)

    for i in range(len(batch_data)):
        x_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        cropped_batch[i, ...] = batch_data[i, ...][x_offset:x_offset+32,
                      y_offset:y_offset+32, :]

        cropped_batch[i, ...] = horizontal_flip(image=cropped_batch[i, ...], axis=1)

    return cropped_batch
    
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
    Xtest = np.load('../../data/cifar10/testData.npy').astype(np.float)
    Ytest = np.load('../../data/cifar10/testLabel.npy').astype(int)

    # padding for data augmentation
    return Xtrain, oneHotRepresentation(Ytrain), Xtest, oneHotRepresentation(Ytest)

def loadDataCifar10_DANN(case):
    Xtrain = np.load('../../data/cifar10/trainData2.npy').astype(np.float)
    Ytrain = np.load('../../data/cifar10/trainLabel2.npy').astype(int)
    Ytest = np.load('../../data/cifar10/testLabel.npy').astype(int)
    if case == 0:
        Xtrain2 = np.load('../../data/cifar10/testData_greyscale.npy').astype(np.float)
        Xtest = np.load('../../data/cifar10/testData_greyscale.npy').astype(np.float)
    elif case == 1:
        Xtrain2 = np.load('../../data/cifar10/testData_negative.npy').astype(np.float)
        Xtest = np.load('../../data/cifar10/testData_negative.npy').astype(np.float)
    elif case == 2:
        Xtrain2 = np.load('../../data/cifar10/testData_randomkernel.npy').astype(np.float)
        Xtest = np.load('../../data/cifar10/testData_randomkernel.npy').astype(np.float)
    elif case == 3:
        Xtrain2 = np.load('../../data/cifar10/testData_radiokernel.npy').astype(np.float)
        Xtest = np.load('../../data/cifar10/testData_radiokernel.npy').astype(np.float)

    # padding for data augmentation
    return Xtrain, Xtrain2, oneHotRepresentation(Ytrain), Xtest, oneHotRepresentation(Ytest)

def load_and_deal(row,column,ngray=16):
    # load data
    Xtrain = np.load('../../data/cifar10/trainData2.npy').astype(np.float)
    Ytrain = np.load('../../data/cifar10/trainLabel2.npy').astype(int)
    Xtest = np.load('../../data/cifar10/testData.npy').astype(np.float)
    Ytest = np.load('../../data/cifar10/testLabel.npy').astype(int)
    Xtrain_gray=tf.image.rgb_to_grayscale(Xtrain)
    Xtest_gray=tf.image.rgb_to_grayscale(Xtest)

    # deal data: get <start pixel> 
    print ("deal date with ngray=%d..." % (ngray))
    direction=np.diag((-1)*np.ones(32*32))
    for i in range(32*32):
        x=int(math.floor(i/32))
        y=int(i%32)
        if x+row<32 and y+column<32:   
            direction[i][i+row*32+column]=1

    xtrain_d=np.copy(Xtrain_gray)
    xtrain_re=np.copy(Xtrain_gray)

    xtest_d=np.copy(Xtest_gray)
    xtest_re=np.copy(Xtest_gray)

    #regularized train_re
    for i in range(xtrain_re.shape[0]):
        xtrain_re[i] = np.asarray(1.0 * xtrain_re[i] * (ngray-1) / xtrain_re[i].max(), dtype=np.int16)
        xtrain_d[i]=np.dot(xtrain_re[i],direction)
    for i in range(xtest_re.shape[0]):
        xtest_re[i] = np.asarray(1.0 * xtest_re[i] * (ngray-1) / xtest_re[i].max(), dtype=np.int16)
        xtest_d[i]=np.dot(xtest_re[i],direction)

    xtr=np.repeat(xtrain_re,ngray,0)
    xtrain_re=xtr.reshape(xtrain_re.shape[0],ngray,32*32)

    xte=np.repeat(xtest_re,ngray,0)
    xtest_re=xte.reshape(xtest_re.shape[0],ngray,32*32) 

    ################# delta ################
    xtr_d=np.repeat(xtrain_d,ngray,0) 
    xtrain_d=xtr_d.reshape(xtrain_d.shape[0],ngray,32*32)
    xte_d=np.repeat(xtest_d,ngray,0)
    xtest_d=xte_d.reshape(xtest_d.shape[0],ngray,32*32)
    ################# delta ################

    return Xtrain, xtrain_re, xtrain_d, ytrain, Xtest, xtest_re, xtest_d, ytest
