import cv2
import math
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
    Xtrain = np.load('data/trainData.npy').astype(np.float)
    Ytrain = np.load('data/trainLabel.npy').astype(int)
    Xtest = np.load('data/testData.npy').astype(np.float)
    Ytest = np.load('data/testLabel.npy').astype(int)

    # padding for data augmentation
    return Xtrain, oneHotRepresentation(Ytrain), Xtest, oneHotRepresentation(Ytest)

def loadDataCifar10_DANN(case):
    Xtrain = np.load('data/trainData.npy').astype(np.float)
    Ytrain = np.load('data/trainLabel.npy').astype(int)
    Ytest = np.load('data/testLabel.npy').astype(int)
    if case == 0:
        Xtrain2 = np.load('data/testData_greyscale.npy').astype(np.float)
        Xtest = np.load('data/testData_greyscale.npy').astype(np.float)
    elif case == 1:
        Xtrain2 = np.load('data/testData_negative.npy').astype(np.float)
        Xtest = np.load('data/testData_negative.npy').astype(np.float)
    elif case == 2:
        Xtrain2 = np.load('data/testData_randomkernel.npy').astype(np.float)
        Xtest = np.load('data/testData_randomkernel.npy').astype(np.float)
    elif case == 3:
        Xtrain2 = np.load('data/testData_radiokernel.npy').astype(np.float)
        Xtest = np.load('data/testData_radiokernel.npy').astype(np.float)

    # padding for data augmentation
    return Xtrain, Xtrain2, oneHotRepresentation(Ytrain), Xtest, oneHotRepresentation(Ytest)

def prepare(img, args):
    row = args.row
    column = args.col
    x = np.copy(img)
    x_d = np.copy(img)
    x_re = np.copy(img)

    x = x.reshape(x.shape[0], 32*32)
    x_re = x_re.reshape(x_re.shape[0], 32*32)
    x_d = x_d.reshape(x_d.shape[0], 32*32)

    direction = np.diag((-1) * np.ones(32*32))
    for i in range(32*32):
        x = int(math.floor(i / 32))
        y = int(i % 32)
        if x + row < 32 and y + column < 32:
            direction[i][i + row * 32 + column] = 1

    for i in range(x_re.shape[0]):
        x_re[i] = np.asarray(1.0 * x_re[i] * (args.ngray - 1) / x_re[i].max(), dtype=np.float32)
        x_d[i] = np.dot(x_re[i], direction)
    return x_d, x_re