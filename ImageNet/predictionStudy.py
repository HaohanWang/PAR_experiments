from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import csv
import time
import json
import argparse
import numpy as np
import math

sys.path.append('../')
import tensorflow as tf
from tensorflow.data import Iterator

from utility.imageLoader import ImageNetLoaderWithDataPath

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from alexNet import AlexNet

def predictionStudy(testFolderPaths, args):
    num_class = 1000

    x = tf.placeholder(tf.float32, (None, 227, 227, 3))
    y = tf.placeholder(tf.float32, (None, num_class))

    # model = AlexNetHex(x, y, x_re, x_d, args, Hex_flag=True)
    model = AlexNet(x, y, args)

    testDataLoader = ImageNetLoaderWithDataPath(batchSize=128, sampleFlag=False)

    with tf.Session() as sess:
        print('Starting testing:')
        sess.run(tf.global_variables_initializer())

        step2load = args.step
        if step2load == -1:
            step2load = None

        model.load_initial_weights(sess, k=step2load)

        start = time.time()

        preds = None
        labels = None
        confidence = None
        paths = []

        for tp in testFolderPaths:
            print("Test path" + tp)
            testDataLoader.setFolderPath(tp)
            test_accuracies = []
            test_k_accuracies = []
            batch_x, batch_y, batch_path = testDataLoader.getNextBatch()
            while batch_x is not None:
                acc, k_accuracy, pred, logits = sess.run([model.accuracy, model.topk_accuracy, model.pred, model.y_conv_loss],
                                           feed_dict={x: batch_x, y: batch_y,
                                                      model.keep_prob: 1.0, model.top_k: 5})
                conf = np.exp(np.max(logits, 1))/np.sum(np.exp(logits), 1)
                if labels is None:
                    labels = np.argmax(batch_y, 1)
                    preds = pred
                    confidence = conf
                else:
                    labels = np.append(labels, np.argmax(batch_y, 1))
                    preds = np.append(preds, pred)
                    confidence = np.append(confidence, conf)
                paths.extend(batch_path)

                test_accuracies.append(acc)
                test_k_accuracies.append(k_accuracy)
                batch_x, batch_y, batch_path = testDataLoader.getNextBatch()
            score = np.mean(test_accuracies)
            k_score = np.mean(test_k_accuracies)

            print("Test Accuracy = %.4f, Top K Accuracy = %.4f, with %.4f minutes passed" % (score, k_score, (time.time()-start)/60.0))
            sys.stdout.flush()

        f = open('filePath.txt', 'w')
        for p in paths:
            f.writelines(p + '\n')
        if step2load is None:
            name2save = 'AlexNet'
        else:
            name2save = 'ALF_'+str(step2load)
        np.save('labels', labels)
        np.save('preds_' + name2save, preds)
        np.save('confidence_' + name2save, confidence)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt_dir', type=str, default='ckpts/', help='Directory for parameter checkpoints')
    parser.add_argument('-l', '--load_params', dest='load_params', action='store_true',
                        help='Restore training from previous model checkpoint?')
    parser.add_argument("-o", "--output", type=str, default='prediction.csv', help='Prediction filepath')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='How many epochs to run in total?')
    parser.add_argument('-b', '--batch_size', type=int, default=128,
                        help='Batch size during training per GPU')
    parser.add_argument('-save', '--save', type=str, default='ckpts/', help='save acc npy path?')
    parser.add_argument('-adv', '--adv_flag', type=int, default=0, help='adversarially training local features')
    parser.add_argument('-m', '--lam', type=float, default=1.0, help='weights of regularization')
    parser.add_argument('-s', '--step', type=int, default=-1, help='with weights we want to load')

    args = parser.parse_args()

    tf.set_random_seed(100)
    np.random.seed()

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    # pretty print args
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))

    testPaths = ['/media/haohanwang/Info/ImageNet/sketch/']
    predictionStudy(testPaths, args)