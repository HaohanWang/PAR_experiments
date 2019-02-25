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
import matplotlib.pyplot as plt
import cv2 
from datagenerator import ImageDataGenerator
from tensorflow.data import Iterator

sys.path.append('../')

#from tensorflow import py_func


import tensorflow as tf

from alex_cnn import MNISTcnn


def set_path(choice):
    
    if choice=='sketch':
        s_tr = 'sourceonly/sketch/train.txt'
        s_val = 'sourceonly/sketch/val.txt'
        s_te='sourceonly/sketch/test.txt'
        return s_tr, s_val, s_te
    if choice=='cartoon':
        c_tr = 'sourceonly/cartoon/train.txt'
        c_val = 'sourceonly/cartoon/val.txt'
        c_te='sourceonly/cartoon/test.txt'
        return c_tr, c_val, c_te
    if choice=='photo':
        p_tr = 'sourceonly/photo/train.txt'
        p_val = 'sourceonly/photo/val.txt'
        p_te='sourceonly/photo/test.txt'
        return p_tr, p_val, p_te
    if choice=='art':
        a_tr = 'sourceonly/art_painting/train.txt'
        a_val = 'sourceonly/art_painting/val.txt'
        a_te='sourceonly/art_painting/test.txt'
        return a_tr, a_val, a_te

    
  
def train(args):
        num_classes = 7
        dataroot = '../../original_data/PACS/'

        cats = ['sketch', 'cartoon', 'photo', 'art']
        cat = cats[args.test]

        batch_size=args.batch_size

        train_file, val_file, test_file=set_path(cat) # can change
        
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

        train_batches_per_epoch = int(np.floor(tr_data.data_size/args.batch_size))
        val_batches_per_epoch = int(np.floor(val_data.data_size / args.batch_size))
        test_batches_per_epoch = int(np.floor(test_data.data_size / args.batch_size))

        x = tf.placeholder(tf.float32,(None,227,227,3))
        y = tf.placeholder(tf.float32, (None, num_classes))
        model = MNISTcnn(x, y, args)
        
        # optimizer = tf.train.AdamOptimizer(1e-4).minimize(model.loss)
        optimizer = tf.train.AdamOptimizer(1e-4) # default was 0.0005
        first_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "cnn")
        first_train_op = optimizer.minimize(model.loss, var_list=first_train_vars)
        if args.adv_flag:
            second_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "adv")
            second_train_op = optimizer.minimize(model.adv_loss, var_list=second_train_vars)

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
            train_acc=[]
            test_acc=[]
            val_acc=[]

            for epoch in range(args.epochs): 
                    
                begin = time.time()
                sess.run(training_init_op)
            
                train_accuracies = []
                train_losses = []
                adv_losses = []
                
                for i in range(train_batches_per_epoch):
                    batch_x, img_batch, batch_y = sess.run(next_batch) 
                   
                    _, acc, loss = sess.run([first_train_op, model.accuracy, model.loss], feed_dict={x: batch_x,
                                                    y: batch_y, 
                                                    model.keep_prob: 0.5, 
                                                    model.e: epoch,
                                                    model.batch: i})
                    if args.adv_flag:
                        _, adv_loss = sess.run([second_train_op, model.adv_loss], feed_dict={x: batch_x, 
                                                    y: batch_y, 
                                                    model.keep_prob: 0.5,
                                                    model.e: epoch,
                                                    model.batch: i})
                        adv_losses.append(adv_loss)
                    train_accuracies.append(acc)
                    train_losses.append(loss)

                train_acc_mean = np.mean(train_accuracies)
                train_acc.append(train_acc_mean)
                train_loss_mean = np.mean(train_losses)

                # compute loss over validation data
                if validation:
                    sess.run(validation_init_op)
                    val_accuracies = []
                    for i in range(val_batches_per_epoch):
                        batch_x, img_batch, batch_y = sess.run(next_batch) 
                        acc = sess.run([model.accuracy], feed_dict={x: batch_x, y: batch_y, 
                                                        model.keep_prob: 1.0, 
                                                        model.e: epoch,
                                                        model.batch: i})
                        val_accuracies.append(acc)

                    val_acc_mean = np.mean(val_accuracies)
                    val_acc.append(val_acc_mean)
                    # log progress to console
                    print("\nEpoch %d, time = %ds, train accuracy = %.4f, loss = %.4f,  validation accuracy = %.4f" % (epoch, time.time()-begin, train_acc_mean,  train_loss_mean, val_acc_mean))
                else:
                    print("\nEpoch %d, time = %ds, train accuracy = %.4f" % (epoch, time.time()-begin, train_acc_mean))
                sys.stdout.flush()

                #test

                if val_acc_mean > best_validate_accuracy:
                    best_validate_accuracy = val_acc_mean

                    test_accuracies = []

                    sess.run(test_init_op)
                    for i in range(test_batches_per_epoch):

                        batch_x, img_batch, batch_y = sess.run(next_batch)
                        acc = sess.run([model.accuracy], feed_dict={x: batch_x, y: batch_y,
                                                        model.keep_prob: 1.0,
                                                        model.e: epoch,
                                                        model.batch: i})
                        test_accuracies.append(acc)

                    score = np.mean(test_accuracies)

                    print("Best Validated Model Prediction Accuracy = %.4f " % (score))

                test_acc.append(score)

                if (epoch + 1) % 10 == 0:
                    ckpt_file = os.path.join(args.ckpt_dir, 'mnist_model.ckpt')

            ckpt_file = os.path.join(args.ckpt_dir, 'mnist_model.ckpt')
            saver.save(sess, ckpt_file)
            """ reuse """  
            print("Best Validated Model Prediction Accuracy = %.4f " % (score))
            return (train_acc,val_acc,test_acc)

def main(args): 
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) 

    train(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt_dir', type=str, default='ckpts/', help='Directory for parameter checkpoints')
    parser.add_argument('-l', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
    parser.add_argument("-o", "--output",  type=str, default='prediction.csv', help='Prediction filepath')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='How many epochs to run in total?')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size during training per GPU') # todo: default was 128
    parser.add_argument('-save','--save',type=str, default='ckpts/', help='save acc npy path?')
    parser.add_argument('-m', '--lam', type=float, default=1.0, help='weights of regularization')
    parser.add_argument('-adv', '--adv_flag', type=int, default=0, help='adversarially training local features')
    parser.add_argument('-test', '--test', type=int, default=0, help='which one to test?')
    parser.add_argument('-g', '--gpu_id', type=str, default='0', help='gpuid used for trianing')

    args = parser.parse_args()

    tf.set_random_seed(100)
    np.random.seed()

    if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    # pretty print args
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) 
    main(args)

   
