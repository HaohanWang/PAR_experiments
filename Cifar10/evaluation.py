import os
import argparse
import numpy as np
import tensorflow as tf
from cnn import ResNet

def oneHotRepresentation(y, num=10):
    r = []
    for i in range(y.shape[0]):
        l = np.zeros(num)
        l[y[i]] = 1
        r.append(l)
    return np.array(r)

def test_cnn(args):
    num_class = 10

    data_path = '../../data/cifar10'
    Ytest = oneHotRepresentation(np.load('../../data/cifar10/testLabel.npy').astype(int))
    domains = ['', '_greyscale', '_negative', '_randomkernel', '_radiokernel']

    args.input_epoch = None

    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.float32, (None, num_class))
    model = ResNet(x, y, args)
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
            print("Mean Accuracy of haohancnn on %s Test Dataset = %.4f " % (domain, score))



def test(args):
    num_class = 10

    data_path = '../../data/cifar10'
    Ytest = oneHotRepresentation(np.load('../../data/cifar10/testLabel.npy').astype(int))
    domains = ['', '_greyscale', '_negative', '_randomkernel', '_radiokernel']

    for i in range(10):
        args.input_epoch = str(i*50)

        tf.reset_default_graph()

        x = tf.placeholder(tf.float32, (None, 32, 32, 3))
        y = tf.placeholder(tf.float32, (None, num_class))
        model = ResNet(x, y, args)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt_dir', type=str, default='ckpts/', help='Directory for parameter checkpoints')
    parser.add_argument('-l', '--load_params', dest='load_params', action='store_true',
                        help='Restore training from previous model checkpoint?')
    parser.add_argument("-o", "--output", type=str, default='cnn', help='Save model filepath')
    parser.add_argument("-i", "--input", type=str, default='haohancnn', help='Load model filepath')
    parser.add_argument("-ie", "--input_epoch", type=int, default=None, help='Load model after n epochs')
    parser.add_argument('-e', '--epochs', type=int, default=1000, help='How many epochs to run in total?')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='Batch size during training per GPU')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed for generating data')
    parser.add_argument('-adv', '--adv_flag', type=int, default=0, help='adversarially training local features')
    parser.add_argument('-m', '--lam', type=float, default=1.0, help='weights of regularization')
    parser.add_argument('-g', '--gpu_id', type=str, default='0', help='gpuid used for trianing')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    # test_cnn(args)
    test(args)
