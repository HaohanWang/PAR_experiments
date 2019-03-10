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
from datagenerator import ImageDataGenerator
from tensorflow.data import Iterator

sys.path.append('../')

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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


def fc(x, num_in, num_out, name, relu=True):
    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    """Create a convolution layer.
    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels / groups,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu


class AlexNet(object):
    def __init__(self, x, y, conf):
        self.x = tf.reshape(x, shape=[-1, 227, 227, 3])
        self.y = y
        self.keep_prob = tf.placeholder(tf.float32)
        self.WEIGHTS_PATH = 'weights/bvlc_alexnet.npy'

        self.class_num = 7

        with tf.variable_scope('cnn'):
            conv1 = conv(self.x, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
            norm1 = lrn(conv1, 2, 1e-05, 0.75, name='norm1')
            pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

            # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
            conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
            norm2 = lrn(conv2, 2, 1e-05, 0.75, name='norm2')
            pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

            # 3rd Layer: Conv (w ReLu)
            conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')

            # 4th Layer: Conv (w ReLu) splitted into two groups
            conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

            # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
            conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
            pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

            # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
            flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
            fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6')
            dropout6 = dropout(fc6, self.keep_prob)

            # 7th Layer: FC (w ReLu) -> Dropout
            fc7 = fc(dropout6, 4096, 4096, name='fc7')

            # fc7 = tf.nn.l2_normalize(fc7, 0)

            dropout7 = dropout(fc7, self.keep_prob)

            h_fc1_drop = dropout7

            # fc2
            with tf.variable_scope("fc2"):
                W_fc2 = weight_variable([4096, self.class_num])
                b_fc2 = bias_variable([self.class_num])
                y_conv_loss = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y_conv_loss))
        self.pred = tf.argmax(y_conv_loss, 1)

        self.correct_prediction = tf.equal(tf.argmax(y_conv_loss, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        if conf.adv_flag:
            [_, m, n, d] = conv5.shape
            with tf.variable_scope('adv'):
                W_a = weight_variable([1, 1, d, self.class_num])
                b_a = bias_variable([self.class_num])
            y_adv_loss = conv2d(conv5, W_a) + b_a
            ty = tf.reshape(self.y, [-1, 1, 1, self.class_num])
            my = tf.tile(ty, [1, m, n, 1])
            self.adv_loss = tf.reduce_min(tf.nn.softmax_cross_entropy_with_logits(labels=my, logits=y_adv_loss))
            # self.adv_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=my, logits=y_adv_loss))
            self.adv_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_adv_loss, -1), tf.argmax(my, -1)), tf.float32))

            self.loss -= conf.lam * self.adv_loss

    def load_initial_weights(self, session):
        """Load weights from file into network.
        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        come as a dict of lists (e.g. weights['conv1'] is a list) and not as
        dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
        'biases') we need a special load function
        """
        # Load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if layer should be trained from scratch
            if op_name != 'fc8':
                with tf.variable_scope('cnn/' + op_name, reuse=True):
                    # Assign weights/biases to their corresponding tf variable
                    for data in weights_dict[op_name]:
                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=True)
                            session.run(var.assign(data))
                        # Weights
                        else:
                            var = tf.get_variable('weights', trainable=True)
                            session.run(var.assign(data))


def oneHotRepresentation(y):
    n = y.shape[0]
    r = np.zeros([n, 7])
    for i in range(r.shape[0]):
        r[int(y[i])] = 1
    return r


def set_path(choice):
    if choice == 'sketch':
        s_tr = 'sourceonly/sketch/train.txt'
        s_val = 'sourceonly/sketch/val.txt'
        s_te = 'sourceonly/sketch/test.txt'
        return s_tr, s_val, s_te
    if choice == 'cartoon':
        c_tr = 'sourceonly/cartoon/train.txt'
        c_val = 'sourceonly/cartoon/val.txt'
        c_te = 'sourceonly/cartoon/test.txt'
        return c_tr, c_val, c_te
    if choice == 'photo':
        p_tr = 'sourceonly/photo/train.txt'
        p_val = 'sourceonly/photo/val.txt'
        p_te = 'sourceonly/photo/test.txt'
        return p_tr, p_val, p_te
    if choice == 'art':
        a_tr = 'sourceonly/art_painting/train.txt'
        a_val = 'sourceonly/art_painting/val.txt'
        a_te = 'sourceonly/art_painting/test.txt'
        return a_tr, a_val, a_te


def train(args):
    num_classes = 7
    dataroot = '../data/PACS/'

    batch_size = args.batch_size

    train_file, val_file, test_file = set_path(args.cat)

    tr_data = ImageDataGenerator(train_file,
                                 dataroot=dataroot,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)
    val_data = ImageDataGenerator(val_file,
                                  dataroot=dataroot,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False)
    test_data = ImageDataGenerator(test_file,
                                   dataroot=dataroot,
                                   mode='inference',
                                   batch_size=batch_size,
                                   num_classes=num_classes,
                                   shuffle=False)

    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()

    training_init_op = iterator.make_initializer(tr_data.data)
    validation_init_op = iterator.make_initializer(val_data.data)
    test_init_op = iterator.make_initializer(test_data.data)

    train_batches_per_epoch = int(np.floor(tr_data.data_size / args.batch_size))
    val_batches_per_epoch = int(np.floor(val_data.data_size / args.batch_size))
    test_batches_per_epoch = int(np.floor(test_data.data_size / args.batch_size))

    x = tf.placeholder(tf.float32, (None, 227, 227, 3))
    y = tf.placeholder(tf.float32, (None, num_classes))
    model = AlexNet(x, y, args)

    optimizer1 = tf.train.AdamOptimizer(1e-5) #1e-5 for art/sketch
    first_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "cnn")
    first_train_op = optimizer1.minimize(model.loss, var_list=first_train_vars)

    if args.adv_flag:
        optimizer2 = tf.train.AdamOptimizer(1e-3)
        second_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "adv")
        second_train_op = optimizer2.minimize(model.adv_loss, var_list=second_train_vars)

    saver = tf.train.Saver(tf.trainable_variables())

    with tf.Session() as sess:

        print('Starting training')
        print('load Alex net weights')

        sess.run(tf.initialize_all_variables())
        model.load_initial_weights(sess)
        if args.load_params:
            ckpt_file = os.path.join(args.ckpt_dir, 'mnist_model.ckpt')
            print('Restoring parameters from', ckpt_file)
            saver.restore(sess, ckpt_file)

        validation = True

        best_validate_accuracy = 0
        score = 0
        train_acc = []
        test_acc = []
        val_acc = []

        for epoch in range(args.epochs):

            begin = time.time()
            sess.run(training_init_op)

            train_accuracies = []
            train_losses = []

            # if args.adv_flag:

            for i in range(train_batches_per_epoch):
                batch_x, img_batch, batch_y = sess.run(next_batch)

                _, acc, loss = sess.run([first_train_op, model.accuracy, model.loss],
                                        feed_dict={x: batch_x, y: batch_y, model.keep_prob: 0.5})
                if args.adv_flag:
                    _, adv_loss = sess.run([second_train_op, model.adv_loss],
                                           feed_dict={x: batch_x, y: batch_y, model.keep_prob: 1.0})

                train_accuracies.append(acc)
                train_losses.append(loss)

            train_acc_mean = np.mean(train_accuracies)
            train_acc.append(train_acc_mean)
            train_loss_mean = np.mean(train_losses)

            # print ()

            # compute loss over validation data
            if validation:
                sess.run(validation_init_op)
                val_accuracies = []
                for i in range(val_batches_per_epoch):
                    batch_x, img_batch, batch_y = sess.run(next_batch)
                    acc = sess.run([model.accuracy], feed_dict={x: batch_x, y: batch_y,
                                                                model.keep_prob: 1.0})
                    val_accuracies.append(acc)

                val_acc_mean = np.mean(val_accuracies)
                val_acc.append(val_acc_mean)
                # log progress to console
                print("Epoch %d, time = %ds, train accuracy = %.4f, loss = %.4f,  validation accuracy = %.4f" % (
                    epoch, time.time() - begin, train_acc_mean, train_loss_mean, val_acc_mean))

                if val_acc_mean > best_validate_accuracy:
                    best_validate_accuracy = val_acc_mean

                    test_accuracies = []

                    sess.run(test_init_op)
                    for i in range(test_batches_per_epoch):
                        batch_x, img_batch, batch_y = sess.run(next_batch)
                        acc = sess.run([model.accuracy], feed_dict={x: batch_x, y: batch_y,
                                                                    model.keep_prob: 1.0})
                        test_accuracies.append(acc)

                    score = np.mean(test_accuracies)

                    print("Best Validated Model Prediction Accuracy = %.4f " % (score))
                sys.stdout.flush()

            test_acc.append(score)

        print("Best Validated Model Prediction Accuracy = %.4f " % (score))
        return (train_acc, val_acc, test_acc)


def main(args):
    train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt_dir', type=str, default='ckpts/', help='Directory for parameter checkpoints')
    parser.add_argument('-l', '--load_params', dest='load_params', action='store_true',
                        help='Restore training from previous model checkpoint?')
    parser.add_argument("-o", "--output", type=str, default='prediction.csv', help='Prediction filepath')
    parser.add_argument('-e', '--epochs', type=int, default=500, help='How many epochs to run in total?')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Batch size during training per GPU')  # todo: default was 128
    parser.add_argument('-save', '--save', type=str, default='ckpts/', help='save acc npy path?')
    parser.add_argument('-cat', '--cat', type=str, default='photo', help='save acc npy path?')
    parser.add_argument('-adv', '--adv_flag', type=int, default=0, help='adversarially training local features')
    parser.add_argument('-m', '--lam', type=float, default=1.0, help='weights of regularization')

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
