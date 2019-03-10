__author__ = 'Haohan Wang'

import cv2
import os
import numpy as np

class ImageNetLoader:
    def __init__(self, batchSize=128, sampleFlag=True):
        text = [line.strip() for line in open('../data/ImageNet/folder2class.txt')]
        self.cl = []
        for line in text:
            items = line.split()
            self.cl.append(items[0])

        self.imgMean = np.array([104.0, 117.0, 124.0], np.float64)

        self.batchSize = batchSize
        self.sampleFlag = sampleFlag
        self.classCounter = -1
        self.batchCounter = -1

        self.data = None
        self.labels = None
        self.reload = True

    def oneHotRepresentation(self, label, k=1000):
        y = np.zeros([label.shape[0], k])
        y[xrange(label.shape[0]), label] = 1
        return y

    def setFolderPath(self, folderPath):
        self.folderPath = folderPath

    def readClassData(self):
        data = []
        labels = []
        for _, _, fl in os.walk(self.folderPath + self.cl[self.classCounter]):
            for f in fl:
                if not f.endswith('.npy'):
                    img = cv2.imread(self.folderPath + self.cl[self.classCounter] + '/' + f)
                    if img is not None:
                        img = cv2.resize(img.astype(float), (227, 227))
                        if len(img.shape) == 2:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                        if img.shape[2] == 3 and img.shape[0] == 227 and img.shape[1] == 227:
                            for i in range(3):
                                img[:,:, i] -= self.imgMean[i]
                            data.append(img)
                            labels.append(self.classCounter)
        self.data = np.array(data)
        self.labels = self.oneHotRepresentation(np.array(labels))
        self.reload = False

    def getNextBatch(self):
        if self.reload:
            self.classCounter += 1
            if self.classCounter == 1000:
                self.classCounter = -1
                return None, None
            else:
                self.readClassData()


        if self.sampleFlag:
            idxs = np.array(np.random.choice(xrange(self.data.shape[0]), size=self.batchSize))
            data = self.data[idxs,:]
            labels = self.labels[idxs,:]
            self.reload = True
            return data, labels
        else:
            self.batchCounter += 1
            data = self.data[self.batchCounter*self.batchSize:(self.batchCounter+1)*self.batchSize, :]
            labels = self.labels[self.batchCounter*self.batchSize:(self.batchCounter+1)*self.batchSize, :]
            if (self.batchCounter+1)*self.batchSize >= self.data.shape[0]:
                self.reload = True
                self.batchCounter = -1
            return data, labels

class ImageNetLoaderWithDataPath:
    def __init__(self, batchSize=128, sampleFlag=True):
        text = [line.strip() for line in open('../data/ImageNet/folder2class.txt')]
        self.cl = []
        for line in text:
            items = line.split()
            self.cl.append(items[0])

        self.imgMean = np.array([104.0, 117.0, 124.0], np.float64)

        self.batchSize = batchSize
        self.sampleFlag = sampleFlag
        self.classCounter = -1
        self.batchCounter = -1

        self.data = None
        self.labels = None
        self.reload = True

    def oneHotRepresentation(self, label, k=1000):
        y = np.zeros([label.shape[0], k])
        y[xrange(label.shape[0]), label] = 1
        return y

    def setFolderPath(self, folderPath):
        self.folderPath = folderPath

    def readClassData(self):
        data = []
        labels = []
        self.paths = []
        for _, _, fl in os.walk(self.folderPath + self.cl[self.classCounter]):
            fl = sorted(fl)
            for f in fl:
                if not f.endswith('.npy'):
                    img = cv2.imread(self.folderPath + self.cl[self.classCounter] + '/' + f)
                    if img is not None:
                        img = cv2.resize(img.astype(float), (227, 227))
                        if len(img.shape) == 2:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                        if img.shape[2] == 3 and img.shape[0] == 227 and img.shape[1] == 227:
                            for i in range(3):
                                img[:,:, i] -= self.imgMean[i]
                            data.append(img)
                            labels.append(self.classCounter)
                            self.paths.append(self.folderPath + self.cl[self.classCounter] + '/' + f)
        self.data = np.array(data)
        self.labels = self.oneHotRepresentation(np.array(labels))
        self.reload = False

    def getNextBatch(self):
        if self.reload:
            self.classCounter += 1
            if self.classCounter == 100:
                self.classCounter = -1
                return None, None, None
            else:
                self.readClassData()


        if self.sampleFlag:
            idxs = np.array(np.random.choice(xrange(self.data.shape[0]), size=self.batchSize))
            data = self.data[idxs,:]
            labels = self.labels[idxs,:]
            self.reload = True
            return data, labels
        else:
            self.batchCounter += 1
            data = self.data[self.batchCounter*self.batchSize:(self.batchCounter+1)*self.batchSize, :]
            labels = self.labels[self.batchCounter*self.batchSize:(self.batchCounter+1)*self.batchSize, :]
            paths = self.paths[self.batchCounter*self.batchSize:(self.batchCounter+1)*self.batchSize]
            if (self.batchCounter+1)*self.batchSize >= self.data.shape[0]:
                self.reload = True
                self.batchCounter = -1
            return data, labels, paths