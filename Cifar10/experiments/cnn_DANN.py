from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import argparse

import numpy as np
import tensorflow as tf

from ..util import nn_util
from ..util import data_util
from ..util import test_util


class ResNet(object):
    def __init__(self, x, x2, y, args):
        self.x = tf.reshape(x, shape=[-1, 32, 32, 3])
        self.x2 = tf.reshape(x2, shape=[-1, 32, 32, 3])
        self.y = y
        self.keep_prob = tf.placeholder(tf.float32)
        self.model_path = os.path.join('../cachedir/models/', args.output)
        self.learning_rate = tf.placeholder(tf.float32)
        self.lamb = tf.placeholder(tf.float32)
        self.load_model_path = os.path.join('../cachedir/models/', args.input, str(args.input_epoch))

        n = 5
        reuse = tf.AUTO_REUSE

        with tf.variable_scope('cnn'):
            layers = []
            layers_2 = []
            with tf.variable_scope('conv0', reuse=reuse):
                conv0 = nn_util.conv_bn_relu_layer(self.x, [3, 3, 3, 16], 1)
                conv0_2 = nn_util.conv_bn_relu_layer(self.x2, [3, 3, 3, 16], 1)
                nn_util.activation_summary(conv0)
                nn_util.activation_summary(conv0_2)
                layers.append(conv0)
                layers_2.append(conv0_2)

            for i in range(n):
                with tf.variable_scope('conv1_%d' % i, reuse=reuse):
                    if i == 0:
                        conv1 = nn_util.residual_block(layers[-1], 16, first_block=True)
                        conv1_2 = nn_util.residual_block(layers_2[-1], 16, first_block=True)
                    else:
                        conv1 = nn_util.residual_block(layers[-1], 16)
                        conv1_2 = nn_util.residual_block(layers_2[-1], 16)
                    nn_util.activation_summary(conv1)
                    nn_util.activation_summary(conv1_2)
                    layers.append(conv1)
                    layers_2.append(conv1_2)

            for i in range(n):
                with tf.variable_scope('conv2_%d' % i, reuse=reuse):
                    conv2 = nn_util.residual_block(layers[-1], 32)
                    conv2_2 = nn_util.residual_block(layers_2[-1], 32)
                    nn_util.activation_summary(conv2)
                    nn_util.activation_summary(conv2_2)
                    layers.append(conv2)
                    layers_2.append(conv2_2)

            for i in range(n):
                with tf.variable_scope('conv3_%d' % i, reuse=reuse):
                    self.conv3 = nn_util.residual_block(layers[-1], 64)
                    self.conv3_2 = nn_util.residual_block(layers_2[-1], 64)
                    layers.append(self.conv3)
                    layers_2.append(self.conv3_2)
                assert self.conv3.get_shape().as_list()[1:] == [8, 8, 64]

            with tf.variable_scope('fc', reuse=reuse):
                in_channel = layers[-1].get_shape().as_list()[-1]
                bn_layer = nn_util.batch_normalization_layer(layers[-1], in_channel)
                relu_layer = tf.nn.relu(bn_layer)
                global_pool = tf.reduce_mean(relu_layer, [1, 2])

                in_channel_2 = layers_2[-1].get_shape().as_list()[-1]
                bn_layer_2 = nn_util.batch_normalization_layer(layers_2[-1], in_channel_2)
                relu_layer_2 = tf.nn.relu(bn_layer_2)
                global_pool_2 = tf.reduce_mean(relu_layer_2, [1, 2])

                assert global_pool.get_shape().as_list()[-1:] == [64]
                output = nn_util.output_layer(global_pool, 10)
                layers.append(output)

        y_conv = output

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y_conv))
        self.pred = tf.argmax(y_conv, 1)

        regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = tf.add_n([self.loss] + regu_losses)

        self.correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        if args.adv_flag:
            with tf.variable_scope('adv', reuse=reuse):
                W_a = nn_util.weight_variable([64, 2])
                b_a = nn_util.bias_variable([2])
                self.y_adv_loss = tf.matmul(global_pool, W_a) + b_a
                self.y_adv_loss_2 = tf.matmul(global_pool_2, W_a) + b_a

            self.adv_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(tf.ones(128, dtype=tf.int32), 2, axis=-1), logits=self.y_adv_loss)) +\
                    tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(tf.zeros(128, dtype=tf.int32), 2, axis=-1), logits=self.y_adv_loss_2))

            self.loss -= self.lamb * self.adv_loss

        optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
        # optimizer = tf.train.AdamOptimizer(self.learning_rate)
        first_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "cnn")
        self.first_train_op = optimizer.minimize(self.loss, var_list=first_train_vars)

        if args.adv_flag:
            optimizer_adv = tf.train.AdamOptimizer(self.learning_rate/10)
            second_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "adv")
            self.second_train_op = optimizer_adv.minimize(self.adv_loss, var_list=second_train_vars)

    def load_initial_weights(self, session):
        for v in tf.trainable_variables():
            saveName = v.name.replace('/', '_')
            # print (saveName)
            if saveName.startswith('cnn'):
                data = np.load(self.load_model_path + '/cnn_' + saveName[4:] + '.npy')
                session.run(v.assign(data))
            elif self.args.input != 'ResNet' and saveName.startswith('adv'):
                data = np.load(self.load_model_path + '/adv_' + saveName[4:] + '.npy')
                session.run(v.assign(data))


def generate_train_batch(args, train_data, train_data2, train_labels, train_batch_size, padding_size, i):
    batch_data = train_data[i*train_batch_size:(i+1)*train_batch_size, :]
    batch_label = train_labels[i*train_batch_size:(i+1)*train_batch_size, :]

    batch_ids = np.random.randint(low=0, high=train_data2.shape[0], size=train_batch_size)
    batch_data2 = train_data2[batch_ids]

    if args.augmentation:
        batch_data = data_util.random_crop_and_flip(batch_data, padding_size=padding_size)
        batch_data2 = data_util.random_crop_and_flip(batch_data2, padding_size=padding_size)

    return batch_data, batch_data2, batch_label

def train(args, Xtrain, X2train, Ytrain, Xtest, Ytest):
    num_class = 10

    model_path = os.path.join('../cachedir/models', args.output)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    x2 = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.float32, (None, num_class))
    model = ResNet(x, x2, y, args)

    with tf.Session() as sess:
        print('Starting training')
        sess.run(tf.global_variables_initializer())
        # model.load_initial_weights(sess)
        num_batches = Xtrain.shape[0] // args.batch_size

        validation = False

        test_num_batches = Xtest.shape[0] // args.batch_size

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
                batch_x, batch_x2, batch_y = generate_train_batch(args, Xtrain, X2train, Ytrain, args.batch_size, 2, i)

                if epoch < args.start_epoch:
                    _, acc, loss = sess.run([model.first_train_op, model.accuracy, model.loss],
                                            feed_dict={x: batch_x, 
                                            x2: batch_x2, 
                                            y: batch_y, 
                                            model.keep_prob: 0.5, 
                                            model.learning_rate: args.learning_rate,
                                            model.lamb: 0})
                else:
                    _, acc, loss = sess.run([model.first_train_op, model.accuracy, model.loss],
                                            feed_dict={x: batch_x, 
                                            x2: batch_x2, 
                                            y: batch_y, 
                                            model.keep_prob: 0.5, 
                                            model.learning_rate: args.learning_rate,
                                            model.lamb: args.lam})
                if args.adv_flag:
                    _, adv_loss = sess.run([model.second_train_op, model.adv_loss],
                                           feed_dict={x: batch_x, 
                                           x2: batch_x2, 
                                           y: batch_y, 
                                           model.keep_prob: 0.5, 
                                           model.learning_rate: args.learning_rate})


                train_accuracies.append(acc)
                losses.append(loss)
                advs.append(adv_loss)

            train_acc_mean = np.mean(train_accuracies)
            train_loss_mean = np.mean(losses)
            adv_loss_mean = np.mean(advs)

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
    parser.add_argument("-o", "--output", type=str, default='dann', help='Save model filepath')
    parser.add_argument("-ie", "--input_epoch", type=str, default=0, help='Load model after n epochs')
    parser.add_argument("-i", "--input", type=str, default='ResNet', help='Load model filepath')
    parser.add_argument('-e', '--epochs', type=int, default=400, help='How many epochs to run in total?')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='Batch size during training per GPU')
    parser.add_argument('-adv', '--adv_flag', type=int, default=1, help='adversarially training local features')
    parser.add_argument('-m', '--lam', type=float, default=1.0, help='weights of regularization')
    parser.add_argument('-g', '--gpu_id', type=str, default='0', help='gpuid used for trianing')
    parser.add_argument('-test', '--test', type=int, default=0, help='which one to test?')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('-au', '--augmentation', type=int, default=1, help='data augmentation?')
    parser.add_argument('-alr', '--adv_learning_rate', type=float, default=1e-3, help='learning rate for adversarial learning')
    parser.add_argument('-se', '--start_epoch', type=int, default=0, help='the epoch start to adversarial training')

    args = parser.parse_args()

    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

    Xtrain, X2train, Ytrain, Xtest, Ytest = data_util.loadDataCifar10_DANN(args.test)
    if args.augmentation:
        pad_width = ((0, 0), (2, 2), (2, 2), (0, 0))
        Xtrain = np.pad(Xtrain, pad_width=pad_width, mode='constant', constant_values=0)
        X2train = np.pad(X2train, pad_width=pad_width, mode='constant', constant_values=0)

    print(Xtrain.shape, Xtest.shape)
    train(args, Xtrain, X2train, Ytrain, Xtest, Ytest)