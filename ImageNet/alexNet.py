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

from utility.imageLoader import ImageNetLoader
from utility.datagenerator import ImageDataGenerator

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
        self.top_k = tf.placeholder(tf.int64)
        self.WEIGHTS_PATH = 'weights/bvlc_alexnet.npy'
        self.NUM_CLASSES = 1000

        with tf.variable_scope('cnn'):
            # conv1
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
            self.rep = fc(dropout6, 4096, 4096, name='fc7')

            # self.rep = tf.nn.l2_normalize(self.rep, 0)

            dropout7 = dropout(self.rep, self.keep_prob)


            # 8th Layer: FC and return unscaled activations
            self.y_conv_loss = fc(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')


        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_conv_loss))
        self.pred = tf.argmax(self.y_conv_loss, 1)

        self.correct_prediction = tf.equal(tf.argmax(self.y_conv_loss, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        topk_correct = tf.nn.in_top_k(self.y_conv_loss, tf.argmax(y, 1), k=self.top_k)
        self.topk_accuracy = tf.reduce_mean(tf.cast(topk_correct, tf.float32))

        if conf.adv_flag:
            [_, m, n, d] = conv1.shape
            with tf.variable_scope('adv'):
                W_a = weight_variable([1, 1, d, self.NUM_CLASSES])
                b_a = bias_variable([self.NUM_CLASSES])

                # with tf.variable_scope('l1'):
                #     W1 = weight_variable([1, 1, d, 100])
                #     b1 = bias_variable([100])
                #     rep1 = tf.nn.relu(conv2d(conv1, W1) + b1)
                #     rep1 = tf.reshape(rep1, [-1, 100])
                # with tf.variable_scope('l2'):
                #     W2 = weight_variable([100, 50])
                #     b2 = bias_variable([50])
                #     rep2 = tf.nn.relu(tf.matmul(rep1, W2) + b2)
                # with tf.variable_scope('l3'):
                #     W3 = weight_variable([50, self.NUM_CLASSES])
                #     b3 = bias_variable([self.NUM_CLASSES])
                #     y_adv_loss = tf.matmul(rep2, W3) + b3
                #     y_adv_loss = tf.reshape(y_adv_loss, [-1, m, n, self.NUM_CLASSES])

            # rep_dropout = dropout(conv1, self.keep_prob)
            y_adv_loss = tf.nn.relu(conv2d(conv1, W_a) + b_a)
            # y_adv_loss = conv2d(conv5, W_a) + b_a

            ty = tf.reshape(self.y, [-1, 1, 1, self.NUM_CLASSES])
            my = tf.tile(ty, [1, m, n, 1])
            self.adv_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=my, logits=y_adv_loss))
            self.adv_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_adv_loss, -1), tf.argmax(my, -1)), tf.float32))

            self.loss -= conf.lam * self.adv_loss

    def load_initial_weights(self, session, k=None):
        """Load weights from file into network.
        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        come as a dict of lists (e.g. weights['conv1'] is a list) and not as
        dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
        'biases') we need a special load function
        """
        # Load the weights into memory
        if k is None:
            weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()
            # Loop over all layer names stored in the weights dict
            for op_name in weights_dict:
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
        else:
            weights_dict = np.load('tuned/weights_'+str(k)+'.npy', encoding='bytes').item()
            # Loop over all layer names stored in the weights dict
            for op_name in weights_dict:
                op_name_str = '/'.join(op_name.split('/')[:-1])
                with tf.variable_scope(op_name_str, reuse=True):
                    # Assign weights/biases to their corresponding tf variable
                    data = weights_dict[op_name]

                    # Biases
                    if len(data.shape) == 1:
                        var = tf.get_variable('biases', trainable=True)
                        session.run(var.assign(data))

                    # Weights
                    else:
                        var = tf.get_variable('weights', trainable=True)
                        session.run(var.assign(data))


def train(args):
    num_classes = 1000

    tr_data = ImageDataGenerator('../data/ImageNet/trainDataPath.txt',
                                 dataroot='/ImageNet/train/',
                                 mode='training',
                                 batch_size=args.batch_size,
                                 num_classes=num_classes,
                                 shuffle=False)
    val_data = ImageDataGenerator('../data/ImageNet/valDataPath.txt',
                                  dataroot='/ImageNet/val/',
                                  mode='inference',
                                  batch_size=args.batch_size,
                                  num_classes=num_classes,
                                  shuffle=False)

    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()

    training_init_op = iterator.make_initializer(tr_data.data)
    validation_init_op = iterator.make_initializer(val_data.data)

    train_batches_per_epoch = int(np.floor(tr_data.data_size / args.batch_size))
    val_batches_per_epoch = int(np.floor(val_data.data_size / args.batch_size))

    x = tf.placeholder(tf.float32, (None, 227, 227, 3))
    y = tf.placeholder(tf.float32, (None, num_classes))

    model = AlexNet(x, y, args)

    optimizer1 = tf.train.AdamOptimizer(1e-5)
    first_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "cnn")
    first_train_op = optimizer1.minimize(model.loss, var_list=first_train_vars)

    if args.adv_flag:
        optimizer2 = tf.train.AdamOptimizer(1e-4)
        second_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "adv")
        second_train_op = optimizer2.minimize(model.adv_loss, var_list=second_train_vars)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Starting training')
        print('load AlexNet weights')
        model.load_initial_weights(sess, k=args.continueEpoch)

        validation = True

        val_acc = []
        for epoch in range(args.epochs):
            if args.continueEpoch is not None and epoch <= args.continueEpoch:
                continue

            begin = time.time()
            sess.run(training_init_op)

            train_accuracies = []
            train_losses = []
            for i in range(train_batches_per_epoch):
                batch_x, img_batch, batch_y = sess.run(next_batch)

                if args.adv_flag:
                    _, adv_loss = sess.run([second_train_op, model.adv_loss],
                                           feed_dict={x: batch_x, y: batch_y, model.keep_prob: 1})

                _, acc, loss = sess.run([first_train_op, model.accuracy, model.loss],
                                        feed_dict={x: batch_x, y: batch_y, model.keep_prob: 0.5, model.top_k: 5})

                train_accuracies.append(acc)
                train_losses.append(loss)

                train_acc_mean = np.mean(train_accuracies[-10:])
                train_loss_mean = np.mean(train_losses[-10:])

                if (i + 1) % 10 == 0:
                    print("Epoch %d, Batch %d/%d, time = %ds, train accuracy = %.4f, loss = %.4f " % (
                        epoch, i + 1, train_batches_per_epoch, time.time() - begin, train_acc_mean, train_loss_mean))

            train_acc_mean = np.mean(train_accuracies)
            train_loss_mean = np.mean(train_losses)

            # compute loss over validation data
            if validation:
                sess.run(validation_init_op)
                val_accuracies = []
                for i in range(val_batches_per_epoch):
                    batch_x, img_batch, batch_y = sess.run(next_batch)
                    acc = sess.run(model.accuracy, feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              model.keep_prob: 1.0, model.top_k: 5})
                    val_accuracies.append(acc)
                val_acc_mean = np.mean(val_accuracies)
                val_acc.append(val_acc_mean)
                # log progress to console
                print("Epoch %d, time = %ds, validation accuracy = %.4f" % (epoch, time.time() - begin, val_acc_mean))
            sys.stdout.flush()


            weights = {}
            for v in tf.trainable_variables():
                weights[v.name] = v.eval()
            np.save('tuned/weights_' + str(epoch), weights)

def test(testFolderPaths, args):
    num_class = 1000

    x = tf.placeholder(tf.float32, (None, 227, 227, 3))
    y = tf.placeholder(tf.float32, (None, num_class))

    model = AlexNet(x, y, args)

    testDataLoader = ImageNetLoader(batchSize=128, sampleFlag=False)

    with tf.Session() as sess:
        print('Starting testing:')
        sess.run(tf.global_variables_initializer())
        model.load_initial_weights(sess, k=args.continueEpoch)

        start = time.time()
        for tp in testFolderPaths:
            print("Test path" + tp)
            testDataLoader.setFolderPath(tp)
            test_accuracies = []
            test_k_accuracies = []
            batch_x, batch_y = testDataLoader.getNextBatch()
            while batch_x is not None:
                acc, k_accuracy = sess.run([model.accuracy, model.topk_accuracy],
                                           feed_dict={x: batch_x, y: batch_y,
                                                      model.keep_prob: 1.0, model.top_k: 5})
                test_accuracies.append(acc)
                test_k_accuracies.append(k_accuracy)
                batch_x, batch_y = testDataLoader.getNextBatch()
            score = np.mean(test_accuracies)
            k_score = np.mean(test_k_accuracies)

            print("Test Accuracy = %.4f, Top K Accuracy = %.4f, with %.4f minutes passed" % (score, k_score, (time.time()-start)/60.0))
            sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt_dir', type=str, default='ckpts/', help='Directory for parameter checkpoints')
    parser.add_argument('-l', '--load_params', dest='load_params', action='store_true',
                        help='Restore training from previous model checkpoint?')
    parser.add_argument("-o", "--output", type=str, default='prediction.csv', help='Prediction filepath')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='How many epochs to run in total?')
    parser.add_argument('-b', '--batch_size', type=int, default=128,
                        help='Batch size during training per GPU') 
    parser.add_argument('-save', '--save', type=str, default='ckpts/', help='save acc npy path?')
    parser.add_argument('-adv', '--adv_flag', type=int, default=0, help='adversarially training local features')
    parser.add_argument('-m', '--lam', type=float, default=1.0, help='weights of regularization')
    parser.add_argument('-p', '--continueEpoch', type=int, default=None, help='which epoch to continue the training')
    parser.add_argument('-t', '--testing', type=int, default=0, help='whether for testing case')

    args = parser.parse_args()

    tf.set_random_seed(100)
    np.random.seed()

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    # pretty print args
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))

    if not args.testing:
        train(args)
    else:
        testPaths = ['/ImageNet/sketch/']
        test(testPaths, args)
