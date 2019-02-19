import numpy as np

trData = np.loadtxt('../data/fashionMNIST/fashion-mnist_train.csv', delimiter=',', skiprows=1)
trlabels = trData[:,0]
trdata = trData[:,1:]


np.save('../data/fashionMNIST/trainData', trdata[:50000,:])
np.save('../data/fashionMNIST/trainLabel', trlabels[:50000])

np.save('../data/fashionMNIST/valData', trdata[50000:,:])
np.save('../data/fashionMNIST/valLabel', trlabels[50000:])

teData = np.loadtxt('../data/fashionMNIST/fashion-mnist_test.csv', delimiter=',', skiprows=1)
telabels = teData[:,0]
tedata = teData[:,1:]


np.save('../data/fashionMNIST/teData', tedata)
np.save('../data/fashionMNIST/teLabel', telabels)