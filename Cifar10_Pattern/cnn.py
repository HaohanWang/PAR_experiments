from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import sys
import csv
import time
import json
import argparse
import numpy as np

sys.path.append('../')

import numpy as np
import tensorflow as tf

from dataLoader import loadMultiDomainCifar10Data, loadDataCifar10


def weight_variable(shape):
    initializer = tf.truncated_normal_initializer(dtype=tf.float32, stddev=5e-2)
    return tf.get_variable("weights", shape, initializer=initializer, dtype=tf.float32)


def bias_variable(shape):
    initializer = tf.constant_initializer(0.0)
    return tf.get_variable("biases", shape, initializer=initializer, dtype=tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')


BN_EPSILON = 0.001


def activation_summary(x):
    '''
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    '''
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''

    ## TODO: to allow different weight decay to fully connected layer and conv layer
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.0002)

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables


def output_layer(input_layer, num_labels):
    '''
    :param input_layer: 2D tensor
    :param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
    :return: output layer Y = WX + B
    '''
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())

    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h


def batch_normalization_layer(input_layer, dimension):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32,
                           initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                            initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer


def conv_bn_relu_layer(input_layer, filter_shape, stride):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''

    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)

    output = tf.nn.relu(bn_layer)
    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''

    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_layer)

    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer


def residual_block(input_layer, output_channel, first_block=False):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block'):
        if first_block:
            filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
            conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)

    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                      input_channel // 2]])
    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output


class ResNet(object):
    def __init__(self, x, y, args):
        self.x = tf.reshape(x, shape=[-1, 32, 32, 3])
        self.y = y
        self.keep_prob = tf.placeholder(tf.float32)
        self.model_path = os.path.join('../../results/Cifar10_Pattern/models/', args.output)
        self.learning_rate = tf.placeholder(tf.float32)

        if args.input_epoch == None:
            self.load_model_path = os.path.join('../../results/Cifar10_Pattern/models/', args.input)
        else:
            self.load_model_path = os.path.join('../../results/Cifar10_Pattern/models/', args.input, args.input_epoch)

        n = 5
        reuse = False
        '''
            The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
            :param input_tensor_batch: 4D tensor
            :param n: num_residual_blocks
            :param reuse: To build train graph, reuse=False. To build validation graph and share weights
            with train graph, resue=True
            :return: last layer in the network. Not softmax-ed
            '''
        with tf.variable_scope('cnn'):

            layers = []
            with tf.variable_scope('conv0', reuse=reuse):
                conv0 = conv_bn_relu_layer(self.x, [3, 3, 3, 16], 1)
                activation_summary(conv0)
                layers.append(conv0)

            for i in range(n):
                with tf.variable_scope('conv1_%d' % i, reuse=reuse):
                    if i == 0:
                        conv1 = residual_block(layers[-1], 16, first_block=True)
                    else:
                        conv1 = residual_block(layers[-1], 16)
                    activation_summary(conv1)
                    layers.append(conv1)

            for i in range(n):
                with tf.variable_scope('conv2_%d' % i, reuse=reuse):
                    conv2 = residual_block(layers[-1], 32)
                    activation_summary(conv2)
                    layers.append(conv2)

            for i in range(n):
                with tf.variable_scope('conv3_%d' % i, reuse=reuse):
                    conv3 = residual_block(layers[-1], 64)
                    layers.append(conv3)
                assert conv3.get_shape().as_list()[1:] == [8, 8, 64]

            with tf.variable_scope('fc', reuse=reuse):
                in_channel = layers[-1].get_shape().as_list()[-1]
                bn_layer = batch_normalization_layer(layers[-1], in_channel)
                relu_layer = tf.nn.relu(bn_layer)
                global_pool = tf.reduce_mean(relu_layer, [1, 2])

                assert global_pool.get_shape().as_list()[-1:] == [64]
                output = output_layer(global_pool, 10)
                layers.append(output)

        y_conv = output

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y_conv))
        self.pred = tf.argmax(y_conv, 1)

        regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = tf.add_n([self.loss] + regu_losses)

        self.correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        if args.adv_flag:
            [_, m, n, d] = conv0.shape
            with tf.variable_scope('adv'):
                W_a = weight_variable([1, 1, d, 10])
                b_a = bias_variable([10])
            y_adv_loss = conv2d(conv0, W_a) + b_a
            ty = tf.reshape(self.y, [-1, 1, 1, 10])
            my = tf.tile(ty, [1, m, n, 1])
            self.adv_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=my, logits=y_adv_loss))
            self.adv_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_adv_loss, -1), tf.argmax(my, -1)), tf.float32))

            self.loss -= args.lam * self.adv_loss

        optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
        # optimizer = tf.train.AdamOptimizer(self.learning_rate)
        first_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "cnn")
        self.first_train_op = optimizer.minimize(self.loss, var_list=first_train_vars)

        if args.adv_flag:
            optimizer_adv = tf.train.AdamOptimizer(1e-3)
            second_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "adv")
            self.second_train_op = optimizer_adv.minimize(self.adv_loss, var_list=second_train_vars)

    def load_initial_weights(self, session):
        for v in tf.trainable_variables():
            saveName = v.name.replace('/', '_')
            # print (saveName)
            if saveName.startswith('cnn'):
                data = np.load(self.load_model_path + '/cnn_' + saveName[4:] + '.npy')
                session.run(v.assign(data))
            elif self.args.input != 'haohancnn' and saveName.startswith('adv'):
                data = np.load(self.load_model_path + '/adv_' + saveName[4:] + '.npy')
                session.run(v.assign(data))

def whitening_image(image_np):
    '''
    Performs per_image_whitening
    :param image_np: a 4D numpy array representing a batch of images
    :return: the image numpy array after whitened
    '''
    for i in range(len(image_np)):
        mean = np.mean(image_np[i, ...])
        # Use adjusted standard deviation here, in case the std == 0.
        std = np.max([np.std(image_np[i, ...]), 1.0/np.sqrt(32 * 32 * 3)])
        image_np[i,...] = (image_np[i, ...] - mean) / std
    return image_np


def horizontal_flip(image, axis):
    '''
    Flip an image at 50% possibility
    :param image: a 3 dimensional numpy array representing an image
    :param axis: 0 for vertical flip and 1 for horizontal flip
    :return: 3D image after flip
    '''
    flip_prop = np.random.randint(low=0, high=2)
    if flip_prop == 0:
        image = cv2.flip(image, axis)

    return image

def random_crop_and_flip(batch_data, padding_size):
    '''
    Helper to random crop and random flip a batch of images
    :param padding_size: int. how many layers of 0 padding was added to each side
    :param batch_data: a 4D batch array
    :return: randomly cropped and flipped image
    '''
    cropped_batch = np.zeros(len(batch_data) * 32 * 32 * 3).reshape(
        len(batch_data), 32, 32, 3)

    for i in range(len(batch_data)):
        x_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        cropped_batch[i, ...] = batch_data[i, ...][x_offset:x_offset+32,
                      y_offset:y_offset+32, :]

        cropped_batch[i, ...] = horizontal_flip(image=cropped_batch[i, ...], axis=1)

    return cropped_batch


def generate_train_batch(args, train_data, train_labels, train_batch_size, padding_size, i):
    '''
    This function helps generate a batch of train data, and random crop, horizontally flip
    and whiten them at the same time
    :param train_data: 4D numpy array
    :param train_labels: 1D numpy array
    :param train_batch_size: int
    :return: augmented train batch data and labels. 4D numpy array and 1D numpy array
    '''
    batch_data = train_data[i*train_batch_size:(i+1)*train_batch_size, :]
    batch_label = train_labels[i*train_batch_size:(i+1)*train_batch_size, :]

    if args.augmentation:
        batch_data = random_crop_and_flip(batch_data, padding_size=padding_size)
        # batch_data = whitening_image(batch_data)

    return batch_data, batch_label

def train(args, Xtrain, Ytrain, Xtest, Ytest):
    num_class = 10

    model_path = os.path.join('../../results/Cifar10_Pattern/models', args.output)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.float32, (None, num_class))
    model = ResNet(x, y, args)

    with tf.Session() as sess:
        print('Starting training')
        sess.run(tf.global_variables_initializer())
        # model.load_initial_weights(sess)
        num_batches = Xtrain.shape[0] // args.batch_size

        validation = False

        test_num_batches = Xtest.shape[0] // args.batch_size

        best_validate_accuracy = 0
        score = 0

        for epoch in range(args.epochs):
            begin = time.time()

            # shuffle the data
            order = np.random.permutation(Xtrain.shape[0])
            Xtrain = Xtrain[order, :]
            Ytrain = Ytrain[order]

            # train
            train_accuracies = []
            losses = []
            advs = []
            adv_loss = 0

            # update the learning rate!
            if epoch == 100 or epoch == 150 or epoch == 200:
                args.learning_rate = 0.1 * args.learning_rate
                print('Learning rate decayed to %.4f'%args.learning_rate)

            for i in range(num_batches):
                batch_x, batch_y = generate_train_batch(args, Xtrain, Ytrain, args.batch_size, 2, i)

                _, acc, loss = sess.run([model.first_train_op, model.accuracy, model.loss],
                                        feed_dict={x: batch_x, 
                                        y: batch_y, 
                                        model.keep_prob: 0.5, 
                                        model.learning_rate: args.learning_rate})
                if args.adv_flag:
                    _, adv_loss = sess.run([model.second_train_op, model.adv_loss],
                                           feed_dict={x: batch_x, 
                                           y: batch_y, 
                                           model.keep_prob: 0.5, 
                                           model.learning_rate: args.learning_rate})


                train_accuracies.append(acc)
                losses.append(loss)
                advs.append(adv_loss)

            train_acc_mean = np.mean(train_accuracies)
            train_loss_mean = np.mean(losses)
            adv_loss_mean = np.mean(advs)

            # print ()

            # compute loss over validation data
            if validation:
                val_num_batches = Xval.shape[0] // args.batch_size
                val_accuracies = []
                for i in range(val_num_batches):
                    batch_x = Xval[i * args.batch_size:(i + 1) * args.batch_size, :]
                    batch_y = Yval[i * args.batch_size:(i + 1) * args.batch_size, :]
                    acc = sess.run(model.accuracy, feed_dict={x: batch_x, 
                                                                y: batch_y, 
                                                                model.keep_prob: 1.0,
                                                                model.learning_rate: args.learning_rate})
                    val_accuracies.append(acc)
                val_acc_mean = np.mean(val_accuracies)

                # log progress to console
                print("Epoch %d, time = %ds, train loss = %.4f, adv_loss = %.4f, train accuracy = %.4f, validation accuracy = %.4f" % (
                epoch, time.time() - begin, train_loss_mean, adv_loss_mean, train_acc_mean, val_acc_mean))
            else:
                print("Epoch %d, time = %ds, train accuracy = %.4f, train_loss_mean=%.4f, adv_loss = %.4f" % (
                    epoch, time.time() - begin, train_acc_mean, train_loss_mean, adv_loss_mean))
            sys.stdout.flush()

            if validation and val_acc_mean > best_validate_accuracy:
                best_validate_accuracy = val_acc_mean

                test_accuracies = []
                for i in range(test_num_batches):
                    batch_x = Xtest[i * args.batch_size:(i + 1) * args.batch_size, :]
                    batch_y = Ytest[i * args.batch_size:(i + 1) * args.batch_size, :]
                    acc = sess.run(model.accuracy, feed_dict={x: batch_x, 
                                                                y: batch_y,
                                                                model.keep_prob: 1.0,
                                                                model.learning_rate: args.learning_rate})
                    test_accuracies.append(acc)
                score = np.mean(test_accuracies)

                print("Best Validated Model Prediction Accuracy = %.4f " % (score))

                for v in tf.trainable_variables():
                    vname = v.name.replace('/', '_')
                    np.save(model_path+'/' + vname, v.eval())
            if (epoch+1)%5==0:
                test_accuracies = []
                for i in range(test_num_batches):
                    batch_x = Xtest[i * args.batch_size:(i + 1) * args.batch_size, :]
                    batch_y = Ytest[i * args.batch_size:(i + 1) * args.batch_size, :]
                    acc = sess.run(model.accuracy, feed_dict={x: batch_x, 
                                                                y: batch_y,
                                                                model.keep_prob: 1.0,
                                                                model.learning_rate: args.learning_rate})
                    test_accuracies.append(acc)

                score = np.mean(test_accuracies)
                print("Epoch %d Prediction Accuracy = %.4f " % (epoch+1, score))
                epoch_model_path = os.path.join(model_path, str(epoch))
                if not os.path.exists(epoch_model_path):
                    os.mkdir(epoch_model_path)
                for v in tf.trainable_variables():
                    vname = v.name.replace('/', '_')
                    np.save(epoch_model_path+'/' + vname, v.eval())

        print("Best Validated Model Prediction Accuracy = %.4f " % (score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load_params', dest='load_params', action='store_true',
                        help='Restore training from previous model checkpoint?')
    parser.add_argument("-o", "--output", type=str, default='cnn', help='Save model filepath')
    parser.add_argument("-ie", "--input_epoch", type=str, default=None, help='Load model after n epochs')
    parser.add_argument("-i", "--input", type=str, default='haohancnn', help='Load model filepath')
    parser.add_argument('-e', '--epochs', type=int, default=500, help='How many epochs to run in total?')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='Batch size during training per GPU')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed for generating data')
    parser.add_argument('-adv', '--adv_flag', type=int, default=0, help='adversarially training local features')
    parser.add_argument('-m', '--lam', type=float, default=1.0, help='weights of regularization')
    parser.add_argument('-g', '--gpu_id', type=str, default='0', help='gpuid used for trianing')
    parser.add_argument('-test', '--test', type=int, default=0, help='which one to test?')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-d', '--dependency', type=int, default=0, help='dependent parttern or independent')
    parser.add_argument('-au', '--augmentation', type=int, default=0, help='data augmentation?')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

    Xtrain, Ytrain, Xtest, Ytest = loadMultiDomainCifar10Data(testCase=args.test, dependency=args.dependency)
    # Xtrain, Ytrain, Xtest, Ytest = loadDataCifar10()
    if args.augmentation:
        # Xtest = whitening_image(Xtest)
        pad_width = ((0, 0), (2, 2), (2, 2), (0, 0))
        Xtrain = np.pad(Xtrain, pad_width=pad_width, mode='constant', constant_values=0)

    print(Xtrain.shape, Xtest.shape)
    train(args, Xtrain, Ytrain, Xtest, Ytest)