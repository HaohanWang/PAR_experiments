__author__ = 'Haohan Wang'

import numpy as np

def run():
    label = np.load('labels.npy')
    filePath = [line.strip() for line in open('filePath.txt')]

    pred1 = np.load('preds_AlexNet.npy')
    pred2 = np.load('preds_ALF_5.npy')

    assert label.shape[0] == len(filePath)
    assert label.shape == pred1.shape

    c = 0
    c1 = 0
    c2 = 0

    print '-----correct ones---------'
    for i in range(len(filePath)):
        if label[i] == pred1[i] and pred1[i] == pred2[i]:
            print filePath[i], '\t', label[i], pred1[i], pred2[i]
            c +=1

    print '-----favoring M1 ---------'
    for i in range(len(filePath)):
        if label[i] == pred1[i] and pred1[i] != pred2[i]:
            print filePath[i], '\t', label[i], pred1[i], pred2[i]
            c1 += 1

    print '-----favoring M2 ---------'
    for i in range(len(filePath)):
        if label[i] == pred2[i] and pred1[i] != pred2[i]:
            print filePath[i], '\t', label[i], pred1[i], pred2[i]
            c2 += 1

    print c, c1, c2

if __name__ == '__main__':
    run()