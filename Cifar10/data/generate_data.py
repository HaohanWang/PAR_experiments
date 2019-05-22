import pickle
import skimage
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def fft(img):
    return np.fft.fft2(img)

def fftshift(img):
    return np.fft.fftshift(fft(img))


def ifft(img):
    return np.fft.ifft2(img)


def ifftshift(img):
    return ifft(np.fft.ifftshift(img))

def distance(i,j,w,h,r):
    dis=np.sqrt((i-16)**2+(j-16)**2)
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

def mask_radial_MM(r=3.5):
    mask = np.zeros((32,32))
    for i in range(32):
        for j in range(32):
            mask[i,j]=distance(i,j,32,32,r=r)
    return [mask, mask, mask]

def mask_random_MM(p=0.5):
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

if __name__ == '__main__':
    print('Loading original Cifar10 dataset...')
    # process and save the training data
    trainX = None
    trainy = None
    for i in range(1, 6):
        filename = 'cifar-10-batches-py/data_batch_%d'%(i)
        batch_data = unpickle(filename)
        if trainX is None:
            trainX = batch_data[b'data'].reshape((-1, 32*32, 3), order='F')
            trainX = trainX.reshape((-1, 32, 32, 3))
            trainy = np.array(batch_data[b'labels'])
        else:
            data = batch_data[b'data'].reshape((-1, 32*32, 3), order='F')
            trainX = np.concatenate((trainX, data.reshape(-1, 32, 32, 3)), axis=0)
            trainy = np.concatenate((trainy, np.array(batch_data[b'labels'])), axis=0)

    np.save('trainData.npy', trainX)
    np.save('trainLabel.npy', trainy)


    # process and save the test data
    filename = 'cifar-10-batches-py/test_batch'
    batch_data = unpickle(filename)
    testX = batch_data[b'data'].reshape((-1, 32*32, 3), order='F')
    testX = testX.reshape((-1, 32, 32, 3))
    testy = np.array(batch_data[b'labels'])
    np.save('testData.npy', testX)
    np.save('testLabel.npy', testy)

    # greyscale domain test data
    print('Processing greyscale Cifar10 images...')
    gs_data = np.zeros(testX.shape, dtype=np.float32)
    for i in range(testX.shape[0]):
        grey_img = skimage.color.rgb2gray(testX[i])
        gs_data[i] = np.tile(grey_img[:, :, np.newaxis], 3)
    np.save('testData_greyscale.npy', gs_data)

    # negative domain test data
    print('Processing negative Cifar10 images...')
    ng_data = 255-testX
    np.save('testData_negative.npy', ng_data)

    # Radio Kernel 
    print('Processing Radio Kernel Cifar10 images...')
    radioMask = mask_radial_MM(r=3.5)
    ro_data = np.zeros(testX.shape, dtype=np.float32)
    for i in range(testX.shape[0]):
        ro_data[i] = addingPattern(testX[i].astype(np.float32), np.abs(radioMask))
    np.save('testData_radiokernel.npy', ro_data)

    # Random Kernel 
    print('Processing Random Kernel Cifar10 images...')
    randomMask = mask_random_MM(p=0.5)
    rm_data = np.zeros(testX.shape, dtype=np.float32)
    for i in range(testX.shape[0]):
        rm_data[i] = addingPattern(testX[i].astype(np.float32), np.abs(randomMask))
    np.save('testData_randomkernel.npy', ro_data)
