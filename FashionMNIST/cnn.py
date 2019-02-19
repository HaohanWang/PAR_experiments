# -*- encoding=utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import csv
import time
import math
import json
import argparse
import numpy as np

sys.path.append('../')

import tensorflow as tf
from utility.dataLoader import loadDataFashionMNIST


def lamda_variable(shape):
    initializer = tf.random_uniform_initializer(dtype=tf.float32, minval=0, maxval=16)
    return tf.get_variable("lamda", shape, initializer=initializer, dtype=tf.float32)


def theta_variable(shape):
    initializer = tf.random_uniform_initializer(dtype=tf.float32, minval=0, maxval=16)
    return tf.get_variable("theta", shape, initializer=initializer, dtype=tf.float32)


def weight_variable(shape):
    initializer = tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1)
    return tf.get_variable("weights", shape, initializer=initializer, dtype=tf.float32)


def bias_variable(shape):
    initializer = tf.constant_initializer(0.0)
    return tf.get_variable("biases", shape, initializer=initializer, dtype=tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class MNISTcnn(object):
    def __init__(self, x, y, conf):
        self.x = tf.reshape(x, shape=[-1, 28, 28, 1])
        self.y = y
        self.keep_prob = tf.placeholder(tf.float32)
        self.class_num = 10

        # conv1
        with tf.variable_scope('cnn'):
            with tf.variable_scope('conv1'):
                W_conv1 = weight_variable([5, 5, 1, 32])
                b_conv1 = bias_variable([32])
                h_conv1 = tf.nn.relu(conv2d(self.x, W_conv1) + b_conv1)
                h_pool1 = max_pool_2x2(h_conv1)

            # conv2
            with tf.variable_scope('conv2'):
                W_conv2 = weight_variable([5, 5, 32, 64])
                b_conv2 = bias_variable([64])
                h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
                h_pool2 = max_pool_2x2(h_conv2)

            # fc1
            with tf.variable_scope("fc1"):
                shape = int(np.prod(h_pool2.get_shape()[1:]))
                W_fc1 = weight_variable([shape, 1024])
                b_fc1 = bias_variable([1024])
                h_pool2_flat = tf.reshape(h_pool2, [-1, shape])
                h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

            h_fc1 = tf.nn.l2_normalize(h_fc1, 0)
            # dropout
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

            # fc2
            with tf.variable_scope("fc2"):
                W_fc2 = weight_variable([1024, self.class_num])
                b_fc2 = bias_variable([self.class_num])
                y_conv_loss = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y_conv_loss))
        self.pred = tf.argmax(y_conv_loss, 1)

        self.correct_prediction = tf.equal(tf.argmax(y_conv_loss, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        if conf.adv_flag:
            self.adv_loss = 0
            self.adv_acc = 0
            [_, m, n, k] = h_pool1.shape
            d = tf.cast(m*n, tf.float32)
            with tf.variable_scope('adv'):
                for i in range(m):
                    for j in range(n):
                        with tf.variable_scope("fc_a_%d_%d" % (i, j)):
                            W_a = weight_variable([k, self.class_num])
                            b_a = bias_variable([self.class_num])
                            y_adv_loss = tf.matmul(h_pool1[:, i, j, :], W_a) + b_a
                            self.adv_loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y_adv_loss)) / d
                            self.adv_acc += tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_adv_loss, 1), tf.argmax(self.y, 1)), tf.float32)) /d

            self.loss -= conf.lam * self.adv_loss


def train(args, Xtrain, Ytrain, Xval, Yval, Xtest, Ytest):
    # """ reuse """
    # with tf.variable_scope('model',reuse=tf.AUTO_REUSE ) as scope:
    num_class = 10

    x = tf.placeholder(tf.float32, (None, 28 * 28))
    y = tf.placeholder(tf.float32, (None, num_class))
    model = MNISTcnn(x, y, args)

    # optimizer = tf.train.AdamOptimizer(1e-4).minimize(model.loss)
    optimizer = tf.train.AdamOptimizer(1e-4)
    first_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "cnn")
    first_train_op = optimizer.minimize(model.loss, var_list=first_train_vars)

    if args.adv_flag:
        second_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "adv")
        second_train_op = optimizer.minimize(model.adv_loss, var_list=second_train_vars)

    saver = tf.train.Saver(tf.trainable_variables())

    with tf.Session() as sess:
        print('Starting training')
        # sess.run(tf.global_variables_initializer())
        sess.run(tf.initialize_all_variables())
        if args.load_params:
            ckpt_file = os.path.join(args.ckpt_dir, 'mnist_model.ckpt')
            print('Restoring parameters from', ckpt_file)
            saver.restore(sess, ckpt_file)

        num_batches = Xtrain.shape[0] // args.batch_size

        validation = True
        val_num_batches = Xval.shape[0] // args.batch_size

        test_num_batches = Xtest.shape[0] // args.batch_size

        best_validate_accuracy = 0
        score = 0
        train_acc = []
        test_acc = []
        val_acc = []
        for epoch in range(args.epochs):

            begin = time.time()

            # train
            ######

            train_accuracies = []
            train_losses = []
            adv_losses = []
            adv_acces = []
            adv_loss = 0
            adv_acc = 0
            for i in range(num_batches):
                batch_x = Xtrain[i * args.batch_size:(i + 1) * args.batch_size, :]
                batch_y = Ytrain[i * args.batch_size:(i + 1) * args.batch_size, :]

                _, acc, loss = sess.run([first_train_op, model.accuracy, model.loss],
                                        feed_dict={x: batch_x, y: batch_y, model.keep_prob: 0.5})
                if args.adv_flag:
                    _, adv_acc, adv_loss = sess.run([second_train_op, model.adv_acc, model.adv_loss],
                                        feed_dict={x: batch_x, y: batch_y, model.keep_prob: 0.5})
                train_accuracies.append(acc)
                train_losses.append(loss)
                adv_losses.append(adv_loss)
                adv_acces.append(adv_acc)
            train_acc_mean = np.mean(train_accuracies)
            train_acc.append(train_acc_mean)

            train_loss_mean = np.mean(train_losses)
            adv_loss_mean = np.mean(adv_losses)
            adv_acc_mean = np.mean(adv_acces)

            # print ()
            # compute loss over validation data
            if validation:
                val_accuracies = []
                for i in range(val_num_batches):
                    batch_x = Xval[i * args.batch_size:(i + 1) * args.batch_size, :]
                    batch_y = Yval[i * args.batch_size:(i + 1) * args.batch_size, :]
                    acc = sess.run(model.accuracy, feed_dict={x: batch_x, y: batch_y, model.keep_prob: 1.0})
                    val_accuracies.append(acc)
                val_acc_mean = np.mean(val_accuracies)
                val_acc.append(val_acc_mean)
                # log progress to console
                print("Epoch %d, time = %ds, train accuracy = %.4f, loss = %.4f, adv loss = %.4f, adv accuracy = %.4f,  validation accuracy = %.4f" % (
                    epoch, time.time() - begin, train_acc_mean, train_loss_mean, adv_loss_mean, adv_acc_mean, val_acc_mean))

                if val_acc_mean > best_validate_accuracy:
                    best_validate_accuracy = val_acc_mean
                    test_accuracies = []
                    for i in range(test_num_batches):
                        batch_x = Xtest[i * args.batch_size:(i + 1) * args.batch_size, :]
                        batch_y = Ytest[i * args.batch_size:(i + 1) * args.batch_size, :]
                        acc = sess.run(model.accuracy, feed_dict={x: batch_x, y: batch_y, model.keep_prob: 1.0})
                        test_accuracies.append(acc)
                    score = np.mean(test_accuracies)

                    print("Best Validated Model Prediction Accuracy = %.4f " % (score))
                test_acc.append(score)

            else:
                print("Epoch %d, time = %ds, train accuracy = %.4f" % (epoch, time.time() - begin, train_acc_mean))
            sys.stdout.flush()

            if (epoch + 1) % 10 == 0:
                ckpt_file = os.path.join(args.ckpt_dir, 'mnist_model.ckpt')
                saver.save(sess, ckpt_file)

        ckpt_file = os.path.join(args.ckpt_dir, 'mnist_model.ckpt')
        saver.save(sess, ckpt_file)
        """ reuse """
        # scope.reuse_variables()
        # draw(train_acc,val_acc,test_acc,corr,args.epochs)

        print("Best Validated Model Prediction Accuracy = %.4f " % (score))
        return (train_acc, val_acc, test_acc)


def main(args):
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = loadDataFashionMNIST()

    # data = input_data.read_data_sets(args.data_dir, one_hot=True, reshape=False, validation_size=args.val_size)

    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))

    train(args, Xtrain, Ytrain, Xval, Yval, Xtest, Ytest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt_dir', type=str, default='ckpts/', help='Directory for parameter checkpoints')
    parser.add_argument('-l', '--load_params', dest='load_params', action='store_true',
                        help='Restore training from previous model checkpoint?')
    parser.add_argument("-o", "--output", type=str, default='prediction.csv', help='Prediction filepath')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='How many epochs to run in total?')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='Batch size during training per GPU')
    parser.add_argument('-save', '--save', type=str, default='ckpts/', help='save acc npy path?')
    parser.add_argument('-adv', '--adv_flag', type=int, default=0, help='adversarially training local features')
    parser.add_argument('-m', '--lam', type=float, default=1.0, help='weights of regularization')

    # print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':')))

    args = parser.parse_args()

    tf.set_random_seed(100)
    np.random.seed()

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    # pretty print args
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))
    main(args)
