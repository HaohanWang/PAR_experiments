__author__ = 'Haohan Wang'

import numpy as np

def run():
    label = np.load('labels.npy')
    filePath = [line.strip() for line in open('filePath.txt')]

    pred1 = np.load('preds_AlexNet.npy')
    pred2 = np.load('preds_ALF_0.npy')

    assert label.shape[0] == len(filePath)
    assert label.shape == pred1.shape

    print '-----correct ones---------'
    for i in range(len(filePath)):
        if label[i] == pred1[i] and pred1[i] == pred2[i]:
            print filePath[i], '\t', label[i], pred1[i], pred2[i]

    print '-----favoring M1 ---------'
    for i in range(len(filePath)):
        if label[i] == pred1[i] and pred1[i] != pred2[i]:
            print filePath[i], '\t', label[i], pred1[i], pred2[i]

    print '-----favoring M2 ---------'
    for i in range(len(filePath)):
        if label[i] == pred2[i] and pred1[i] != pred2[i]:
            print filePath[i], '\t', label[i], pred1[i], pred2[i]

if __name__ == '__main__':
    run()