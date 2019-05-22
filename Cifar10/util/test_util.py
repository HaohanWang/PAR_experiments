import os
import numpy as np
import tensorflow as tf
from data_util import prepare

def oneHotRepresentation(y, num=10):
    r = []
    for i in range(y.shape[0]):
        l = np.zeros(num)
        l[y[i]] = 1
        r.append(l)
    return np.array(r)


def generate_test_batch(args, test_data, test_labels, test_batch_size, padding_size, i):
    batch_data = test_data[i*test_batch_size:(i+1)*test_batch_size, :]
    batch_label = test_labels[i*test_batch_size:(i+1)*test_batch_size, :]

    gray=np.dot(batch_data[...,:3], [0.2989, 0.5870, 0.1140])

    return batch_data, batch_label, gray


def test(args, model):
    num_class = 10

    data_path = '../data/cifar10'
    Ytest = oneHotRepresentation(np.load('../data/cifar10/testLabel.npy').astype(int))
    domains = ['_greyscale', '_negative', '_randomkernel', '_radiokernel']

    tf.reset_default_graph()
    args.input = args.output
    args.input_epoch = 399
    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.float32, (None, num_class))
    with tf.Session() as sess:
        print('Starting Evaluation')
        sess.run(tf.global_variables_initializer())
        model.load_initial_weights(sess)

        for domain in domains:
            test_file_path = os.path.join(data_path, 'testData%s.npy'%(domain))
            Xtest = np.load(test_file_path)
            test_num_batches = Xtest.shape[0] // args.batch_size

            test_accuracies = []
            for i in range(test_num_batches):
                batch_x = Xtest[i * args.batch_size:(i + 1) * args.batch_size, :]
                batch_y = Ytest[i * args.batch_size:(i + 1) * args.batch_size, :]
                acc = sess.run(model.accuracy, feed_dict={x: batch_x, y: batch_y, model.keep_prob: 1.0})
                test_accuracies.append(acc)
            score = np.mean(test_accuracies)
            print("Mean Accuracy of epoch %d on %s Test Dataset = %.4f " % (int(args.input_epoch), domain, score))


def test_HEX(args):
    num_class = 10

    data_path = '../../data/cifar10'
    Ytest = oneHotRepresentation(np.load('../../data/cifar10/testLabel.npy').astype(int))
    domains = ['_greyscale', '_negative', '_randomkernel', '_radiokernel']

    tf.reset_default_graph()
    args.input = args.output
    args.input_epoch = 399
    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.float32, (None, num_class))
    x_re = tf.placeholder(tf.float32, (None, 32 * 32))
    x_d = tf.placeholder(tf.float32, (None, 32 * 32))
    with tf.Session() as sess:
        print('Starting Evaluation')
        sess.run(tf.global_variables_initializer())
        model.load_initial_weights(sess)

        for domain in domains:
            test_file_path = os.path.join(data_path, 'testData%s.npy'%(domain))
            Xtest = np.load(test_file_path)
            test_num_batches = Xtest.shape[0] // args.batch_size

            test_accuracies = []
            for i in range(test_num_batches):
                batch_x, batch_y, img_batch = generate_test_batch(args, Xtest, Ytest, args.batch_size, 2, i)
                batch_xd, batch_re = prepare(img_batch, args)
                p, acc = sess.run([model.pred, model.accuracy], feed_dict={x: batch_x, 
                                                        x_re: batch_re,
                                                        x_d: batch_xd, 
                                                        y: batch_y, 
                                                        model.keep_prob: 1.0})
                test_accuracies.append(acc)
            score = np.mean(test_accuracies)
            print("Mean Accuracy of epoch %d on %s Test Dataset = %.4f " % (int(args.input_epoch), domain, score))