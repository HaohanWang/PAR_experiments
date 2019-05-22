from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import sys
import csv
import math
import time
import json
import argparse
import numpy as np

sys.path.append('../')

import numpy as np
import tensorflow as tf

from dataLoader import loadDataCifar10_2

def lamda_variable(shape): # NGLCM related
    initializer = tf.random_uniform_initializer(dtype=tf.float32, minval=0, maxval=shape[0])
    return tf.get_variable("lamda", shape, initializer=initializer, dtype=tf.float32)

def theta_variable(shape): # NGLCM related
    initializer = tf.random_uniform_initializer(dtype=tf.float32, minval=0, maxval=shape[0])
    return tf.get_variable("theta", shape, initializer=initializer, dtype=tf.float32)

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
    def __init__(self, x, y, x_re, x_d, args, Hex_flag=False):
        self.x = tf.reshape(x, shape=[-1, 32, 32, 3])
        self.x_re = tf.reshape(x_re, shape=[-1, 1, 1024])
        self.x_d = tf.reshape(x_d, shape=[-1, 1, 1024])
        self.y = y
        self.keep_prob = tf.placeholder(tf.float32)
        self.model_path = os.path.join('../../results/Cifar10/models/', args.output)
        self.learning_rate = tf.placeholder(tf.float32)

        if int(args.input_epoch) == 0:
            self.load_model_path = os.path.join('../../results/Cifar10/models/', args.input)
        else:
            self.load_model_path = os.path.join('../../results/Cifar10/models/', args.input, str(args.input_epoch))

        # --------------------------

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
                # B x 64
                global_pool = tf.reduce_mean(relu_layer, [1, 2])
                assert global_pool.get_shape().as_list()[-1:] == [64]

        with tf.variable_scope('fc2', reuse=reuse):
            # NGLCM
            with tf.variable_scope('nglcm'):
                self.lamda = lamda_variable([args.ngray, 1])
                theta = theta_variable([args.ngray, 1])
                self.g = tf.matmul(tf.minimum(tf.maximum(tf.subtract(self.x_d, self.lamda), 1e-5), 1),
                              tf.minimum(tf.maximum(tf.subtract(self.x_re, theta), 1e-5), 1), transpose_b=True)/1024.0

            with tf.variable_scope("nglcm_fc1"):
                g_flat = tf.reshape(self.g, [-1, args.ngray * args.ngray])
                glgcm_W_fc1 = weight_variable([args.ngray * args.ngray, 32])
                glgcm_b_fc1 = bias_variable([32])
                self.glgcm_h_fc1 = tf.nn.relu(tf.matmul(g_flat, glgcm_W_fc1) + glgcm_b_fc1)

            # concatenate the representations (Equation 3 in paper)
            yconv_contact_loss = tf.concat([global_pool, self.glgcm_h_fc1], 1)
            pad = tf.zeros_like(self.glgcm_h_fc1, tf.float32)
            yconv_contact_pred = tf.concat([global_pool, pad], 1)
            pad2 = tf.zeros_like(global_pool, tf.float32)
            yconv_contact_H = tf.concat([pad2, self.glgcm_h_fc1], 1)

            # --------------------------
            input_dim = yconv_contact_loss.get_shape().as_list()[-1]
            W_fc2 = weight_variable([input_dim, 10])
            b_fc2 = bias_variable([10])
            y_conv_loss = tf.matmul(yconv_contact_loss, W_fc2) + b_fc2
            y_conv_pred = tf.matmul(yconv_contact_pred, W_fc2) + b_fc2
            self.y_conv_H = tf.matmul(yconv_contact_H, W_fc2) + b_fc2
            self.interest = tf.matmul(self.y_conv_H, self.y_conv_H, transpose_a=True)[:2]

            layers.append(y_conv_loss)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y_conv_loss))
        self.pred = tf.argmax(y_conv_pred, 1)

        if Hex_flag:
            # Projection (Equation 4 in the paper)
            # Notice that, we are using the most succinct form of HEX as an example
            y_conv_loss = y_conv_loss - tf.matmul(tf.matmul(tf.matmul(self.y_conv_H, tf.matrix_inverse(tf.matmul(self.y_conv_H, self.y_conv_H, transpose_a=True)+tf.eye(self.y_conv_H.get_shape().as_list()[-1]))),
                              self.y_conv_H, transpose_b=True), y_conv_loss)
            # --------------------------

            # # another form that involves a hyperparameter which can help the superficial statistics learner to summarize related statistics
            # # we noticed that this form does not contribute much when the superficial statistics learner is NGLCM, but can be helpful in other cases
            # y_conv_loss = y_conv_loss - tf.matmul(tf.matmul(tf.matmul(y_conv_H, tf.matrix_inverse(tf.matmul(y_conv_H, y_conv_H, transpose_a=True))),
            #                   y_conv_H, transpose_b=True), y_conv_loss) \
            #               + self.lam * y_conv_H
            # # --------------------------

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y_conv_loss))

        regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = tf.add_n([self.loss] + regu_losses)

        self.correct_prediction = tf.equal(tf.argmax(y_conv_pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc2")
        print(self.train_vars)
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9).minimize(self.loss, var_list=self.train_vars)

        self.d_clip = [v.assign(tf.clip_by_value(v, -args.ngray, args.ngray)) for v in self.train_vars]

    def load_initial_weights(self, session):
        for v in tf.trainable_variables():
            saveName = v.name.replace('/', '_')
            print(saveName)
            if saveName.startswith('fc2'):
                data = np.load(self.load_model_path + '/fc2_' + saveName[4:] + '.npy')
                session.run(v.assign(data))
            if saveName.startswith('cnn'):
                data = np.load('/home/songweig/AdvLF/results/Cifar10/models/resnet/199' + '/cnn_' + saveName[4:] + '.npy')
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

def preparion(img, args):
    row = args.row
    column = args.col
    x = np.copy(img)
    x_d = np.copy(img)
    x_re = np.copy(img)

    x = x.reshape(x.shape[0], 32*32)
    x_re = x_re.reshape(x_re.shape[0], 32*32)
    x_d = x_d.reshape(x_d.shape[0], 32*32)

    direction = np.diag((-1) * np.ones(32*32))
    for i in range(32*32):
        x = int(math.floor(i / 32))
        y = int(i % 32)
        if x + row < 32 and y + column < 32:
            direction[i][i + row * 32 + column] = 1

    for i in range(x_re.shape[0]):
        x_re[i] = np.asarray(1.0 * x_re[i] * (args.ngray - 1) / x_re[i].max(), dtype=np.float32)
        x_d[i] = np.dot(x_re[i], direction)
    return x_d, x_re

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

    gray=np.dot(batch_data[...,:3], [0.2989, 0.5870, 0.1140])

    return batch_data, batch_label, gray

def generate_test_batch(args, test_data, test_labels, test_batch_size, padding_size, i):
    batch_data = test_data[i*test_batch_size:(i+1)*test_batch_size, :]
    batch_label = test_labels[i*test_batch_size:(i+1)*test_batch_size, :]

    gray=np.dot(batch_data[...,:3], [0.2989, 0.5870, 0.1140])

    return batch_data, batch_label, gray

def train(args, Xtrain, Ytrain, Xtest, Ytest):
    num_class = 10

    model_path = os.path.join('../../results/Cifar10/models', args.output)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.float32, (None, num_class))
    x_re = tf.placeholder(tf.float32, (None, 32 * 32))
    x_d = tf.placeholder(tf.float32, (None, 32 * 32))
    model = ResNet(x, y, x_re, x_d, args, Hex_flag=True)

    with tf.Session() as sess:
        print('Starting training')
        sess.run(tf.global_variables_initializer())
        model.load_initial_weights(sess)
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

            # update the learning rate!
            if epoch == 100 or epoch == 150 or epoch == 200:
                args.learning_rate = 0.1 * args.learning_rate
                print('Learning rate decayed to %.4f'%args.learning_rate)

            for i in range(num_batches):
                batch_x, batch_y, img_batch = generate_train_batch(args, Xtrain, Ytrain, args.batch_size, 2, i)
                batch_xd, batch_re = preparion(img_batch, args)
                # sess.run(model.d_clip)
                # interesed_var = sess.run([model.interest], feed_dict={x: batch_x,
                #                                                     x_re: batch_re,
                #                                                     x_d: batch_xd})

                # print(interesed_var)
                _, acc, loss = sess.run([model.optimizer, model.accuracy, model.loss], feed_dict={x: batch_x,
                                                                                            x_re: batch_re,
                                                                                            x_d: batch_xd,
                                                                                            y: batch_y,
                                                                                            model.keep_prob: 0.5,
                                                                                            model.learning_rate: args.learning_rate})
                # print(batch_xd)
                # print(interesed_var)

                # clip the weights

                train_accuracies.append(acc)
                losses.append(loss)

            train_acc_mean = np.mean(train_accuracies)
            train_loss_mean = np.mean(losses)

            print("Epoch %d, time = %ds, train accuracy = %.4f, train_loss_mean=%.4f" % (
                epoch, time.time() - begin, train_acc_mean, train_loss_mean,))
            sys.stdout.flush()

            if (epoch+1)%5==0:
                test_accuracies = []
                for i in range(test_num_batches):
                    batch_x, batch_y, img_batch = generate_test_batch(args, Xtest, Ytest, args.batch_size, 2, i)
                    batch_xd, batch_re = preparion(img_batch, args)

                    acc = sess.run(model.accuracy, feed_dict={x: batch_x,
                                                                x_re: batch_re,
                                                                x_d: batch_xd,
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
    parser.add_argument("-o", "--output", type=str, default='hex', help='Save model filepath')
    parser.add_argument("-ie", "--input_epoch", type=str, default=199, help='Load model after n epochs')
    parser.add_argument("-i", "--input", type=str, default='resnet', help='Load model filepath')
    parser.add_argument('-e', '--epochs', type=int, default=400, help='How many epochs to run in total?')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='Batch size during training per GPU')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed for generating data')
    parser.add_argument('-g', '--gpu_id', type=str, default='0', help='gpuid used for trianing')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('-au', '--augmentation', type=int, default=1, help='data augmentation?')
    parser.add_argument('-row', '--row', type=int, default=0, help='direction delta in row')
    parser.add_argument('-col', '--col', type=int, default=0, help='direction delta in column')
    parser.add_argument('-ng', '--ngray', type=int, default=16, help='regularization gray level')
    args = parser.parse_args()

    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

    Xtrain,Ytrain, Xtest, Ytest = loadDataCifar10_2()
    if args.augmentation:
        # Xtest = whitening_image(Xtest)
        pad_width = ((0, 0), (2, 2), (2, 2), (0, 0))
        Xtrain = np.pad(Xtrain, pad_width=pad_width, mode='constant', constant_values=0)

    print(Xtrain.shape, Xtest.shape)
    train(args, Xtrain, Ytrain, Xtest, Ytest)