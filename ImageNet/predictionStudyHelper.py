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

def details():
    label = np.load('labels.npy')
    filePath = [line.strip() for line in open('filePath.txt')]

    labels = [line.strip().split()[1] for line in open('../data/ImageNet/folder2class.txt')]

    pred1 = np.load('preds_AlexNet.npy')
    pred2 = np.load('preds_ALF_5.npy')
    conf1 = np.load('confidence_AlexNet.npy')
    conf2 = np.load('confidence_ALF_5.npy')

    assert label.shape[0] == len(filePath)
    assert label.shape == pred1.shape


    favor1 = {}

    for i in range(len(filePath)):
        if label[i] == pred1[i] and pred1[i] != pred2[i]:
            # print filePath[i], '\t', label[i], pred1[i], pred2[i]
            if label[i] not in favor1:
                favor1[label[i]] = {}
            if pred2[i] not in favor1[label[i]]:
                favor1[label[i]][pred2[i]] = 0
            favor1[label[i]][pred2[i]] += 1


    favor2 = {}

    for i in range(len(filePath)):
        if label[i] == pred2[i] and pred1[i] != pred2[i]:
            # print filePath[i], '\t', label[i], pred1[i], pred2[i]
            if label[i] not in favor2:
                favor2[label[i]] = {}
            if pred1[i] not in favor2[label[i]]:
                favor2[label[i]][pred1[i]] = 0
            favor2[label[i]][pred1[i]] += 1


    # print '-----favoring M1 ---------'
    # for i in range(len(filePath)):
    #     if label[i] == pred1[i] and pred1[i] != pred2[i]:
    #         if favor1[label[i]][pred2[i]] >= 3:
    #             flag = True
    #             if label[i] in favor2 and pred2[i] in favor2[label[i]]:
    #                 flag = False
    #             if pred2[i] in favor2 and label[i] in favor2[pred2[i]]:
    #                 flag = False
    #             if flag:
    #                 print filePath[i], '\t\t', label[i],  pred1[i], pred2[i], '\t', labels[label[i]], labels[pred2[i]]

    print '-----favoring M2 ---------'
    for i in range(len(filePath)):
        if label[i] == pred2[i] and pred1[i] != pred2[i]:
            if favor2[label[i]][pred1[i]] >= 3:
                flag = True
                if label[i] in favor1 and pred1[i] in favor1[label[i]]:
                    flag = False
                if pred1[i] in favor1 and label[i] in favor1[pred1[i]]:
                    flag = False
                if flag:
                    print filePath[i], '\t\t', label[i],  pred1[i], pred2[i], '\t', labels[label[i]], labels[pred1[i]], '\t', conf2[i], conf1[i]



if __name__ == '__main__':
    details()