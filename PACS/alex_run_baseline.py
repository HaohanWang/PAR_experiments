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
from tensorflow.contrib.data import Iterator

sys.path.append('../')

#from tensorflow import py_func


import tensorflow as tf

from alex_cnn_baseline import MNISTcnn


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

    
  
def train(args, use_hex=True):
        num_classes = 7
        dataroot = '../data/PACS/'

        cat = 'photo'

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
        model = MNISTcnn(x, y, args, Hex_flag=use_hex)
        
        # optimizer = tf.train.AdamOptimizer(1e-4).minimize(model.loss)
        optimizer = tf.train.AdamOptimizer(1e-4) # default was 0.0005
        first_train_op = optimizer.minimize(model.loss)
        
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

            val_rep = None
            val_re = None
            val_d = None
            val_y = None

            for epoch in range(args.epochs): 
                    
                begin = time.time()
                sess.run(training_init_op)
                #sess.run(validation_init_op)
                #sess.run(test_init_op)
                # train
                ######
            
                train_accuracies = []
                train_losses = []
                train_rep = None
                train_re = None
                train_d = None
                train_y = None
                for i in range(train_batches_per_epoch):
                    batch_x, img_batch, batch_y = sess.run(next_batch) 
                    batch_xd,batch_re=preparion(img_batch,args)
                   
                    _, acc, loss, rep = sess.run([first_train_op, model.accuracy, model.loss, model.rep], feed_dict={x: batch_x,
                                                    x_re: batch_re, 
                                                    x_d: batch_xd, 
                                                    y: batch_y, 
                                                    model.keep_prob: 0.5, 
                                                    model.e: epoch,
                                                    model.batch: i})
                   
                    train_accuracies.append(acc)
                    train_losses.append(loss)

                    if train_rep is None:
                        train_rep = rep
                    else:
                        train_rep = np.append(train_rep, rep, 0)

                    if train_re is None:
                        train_re = batch_re
                    else:
                        train_re = np.append(train_re, batch_re, 0)

                    if train_d is None:
                        train_d = batch_xd
                    else:
                        train_d = np.append(train_d, batch_xd, 0)

                    if train_y is None:
                        train_y = batch_y
                    else:
                        train_y = np.append(train_y, batch_y, 0)

                train_acc_mean = np.mean(train_accuracies)
                train_acc.append(train_acc_mean)
                train_loss_mean = np.mean(train_losses)

                # print ()

                # compute loss over validation data
                if validation:
                    sess.run(validation_init_op)
                    val_accuracies = []
                    val_rep = None
                    val_re = None
                    val_d = None
                    val_y = None
                    for i in range(val_batches_per_epoch):
                        batch_x, img_batch, batch_y = sess.run(next_batch) 
                        batch_xd,batch_re=preparion(img_batch,args)
                        acc, rep = sess.run([model.accuracy, model.rep], feed_dict={x: batch_x, x_re:batch_re,
                                                        x_d: batch_xd, y: batch_y, 
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

                    test_rep = None
                    test_re = None
                    test_d = None
                    test_y = None

                    sess.run(test_init_op)
                    for i in range(test_batches_per_epoch):

                        batch_x, img_batch, batch_y = sess.run(next_batch)
                        batch_xd, batch_re=preparion(img_batch,args)
                        acc, rep = sess.run([model.accuracy, model.rep], feed_dict={x: batch_x,
                                                        x_re: batch_re, x_d: batch_xd, y: batch_y,
                                                        model.keep_prob: 1.0,
                                                        model.e: epoch,
                                                        model.batch: i})
                        test_accuracies.append(acc)

                        if test_rep is None:
                            test_rep = rep
                        else:
                            test_rep = np.append(test_rep, rep, 0)

                        if test_re is None:
                            test_re = batch_re
                        else:
                            test_re = np.append(test_re, batch_re, 0)

                        if test_d is None:
                            test_d = batch_xd
                        else:
                            test_d = np.append(test_d, batch_xd, 0)

                        if test_y is None:
                            test_y = batch_y
                        else:
                            test_y = np.append(test_y, batch_y, 0)

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

    if args.hex==1:
        (h_train_acc,h_val_acc,h_test_acc)=train(args, True)
        hex_acc=np.array((h_train_acc,h_val_acc,h_test_acc))
        np.save(args.save+'hex_acc_'+str(args.corr)+'_'+str(args.row)+'_'+str(args.col)+'_'+str(args.div)+'.npy',hex_acc)
    else:
        (n_train_acc,n_val_acc,n_test_acc)=train(args, False)
        acc=np.array((n_train_acc,n_val_acc,n_test_acc))
        np.save(args.save+'acc_'+str(args.corr)+'_'+str(args.row)+'_'+str(args.col)+'_'+str(args.div)+'.npy',acc)
    #draw_all(h_train_acc,h_val_acc,h_test_acc,n_train_acc,n_val_acc,n_test_acc,corr)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt_dir', type=str, default='ckpts/', help='Directory for parameter checkpoints')
    parser.add_argument('-l', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
    parser.add_argument("-o", "--output",  type=str, default='prediction.csv', help='Prediction filepath')
    parser.add_argument('-e', '--epochs', type=int, default=25000, help='How many epochs to run in total?')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size during training per GPU') # todo: default was 128
    parser.add_argument('-re', '--re', type=int, default=0, help='regularization?')
    parser.add_argument('-corr', '--corr', type=int, default=8, help='correlation')
    parser.add_argument('-hex','--hex',type=int, default=1, help='use hex?')
    parser.add_argument('-save','--save',type=str, default='hex2/', help='save acc npy path?')
    parser.add_argument('-row', '--row', type=int, default=0, help='direction delta in row')
    parser.add_argument('-col', '--col', type=int, default=0, help='direction delta in column')
    parser.add_argument('-ng', '--ngray', type=int, default=16, help='regularization gray level')
    parser.add_argument('-div', '--div', type=int, default=200, help='how many epochs before HEX start')
    #print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) 

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

   
