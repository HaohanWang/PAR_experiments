# change dataloader to load npy with corr=0.0~1.0
import numpy as np

def loadDataSentiment(corr=0.8):
    dataPath = '../../original_data/background_npy/npy_'+str(int(corr*10))+'/'
    #dataPath = '../../../data/background_npy/npy_'+str(int(corr*10))+'/'
    # np.random.seed(0)
    Xtrain = np.load(dataPath + 'trainData_small.npy').astype(np.float32)
    # np.random.shuffle(Xtrain)
    Xval = np.load(dataPath + 'valData_small.npy').astype(np.float32)
    Xtest = np.load(dataPath + 'testData_small.npy').astype(np.float32)
    Ytrain = np.load(dataPath + 'trainLabel_small_onehot.npy').astype(np.float32)
    Yval = np.load(dataPath + 'valLabel_small_onehot.npy').astype(np.float32)
    Ytest = np.load(dataPath + 'testLabel_small_onehot.npy').astype(np.float32)
   
    return Xtrain, Ytrain, Xval, Yval, Xtest, Ytest
