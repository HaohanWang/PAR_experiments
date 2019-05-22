import os
import math
import argparse
import numpy as np
import tensorflow as tf
from cnn_hex import ResNet

def oneHotRepresentation(y, num=10):
    r = []
    for i in range(y.shape[0]):
        l = np.zeros(num)
        l[y[i]] = 1
        r.append(l)
    return np.array(r)

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

def generate_test_batch(args, test_data, test_labels, test_batch_size, padding_size, i):
    batch_data = test_data[i*test_batch_size:(i+1)*test_batch_size, :]
    batch_label = test_labels[i*test_batch_size:(i+1)*test_batch_size, :]

    gray=np.dot(batch_data[...,:3], [0.2989, 0.5870, 0.1140])

    return batch_data, batch_label, gray

def test(args):
    num_class = 10

    data_path = '../../data/cifar10'
    Ytest = oneHotRepresentation(np.load('../../data/cifar10/testLabel.npy').astype(int))
    domains = ['_randomkernel']
    # domains = ['', '_greyscale', '_negative', '_randomkernel', '_radiokernel', '_semanticadv', '_semanticadv_resnet']

    for i in range(20):
        tf.reset_default_graph()
        args.input_epoch = (i+1)*5-1
        x = tf.placeholder(tf.float32, (None, 32, 32, 3))
        y = tf.placeholder(tf.float32, (None, num_class))
        x_re = tf.placeholder(tf.float32, (None, 32 * 32))
        x_d = tf.placeholder(tf.float32, (None, 32 * 32))
        model = ResNet(x, y, x_re, x_d, args, Hex_flag=True)
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
                    batch_xd, batch_re = preparion(img_batch, args)
                    p, acc = sess.run([model.pred, model.accuracy], feed_dict={x: batch_x, 
                                                            x_re: batch_re,
                                                            x_d: batch_xd, 
                                                            y: batch_y, 
                                                            model.keep_prob: 1.0})
                    test_accuracies.append(acc)
                    print(p[:10])
                score = np.mean(test_accuracies)
                print("Mean Accuracy of epoch %d on %s Test Dataset = %.4f " % (int(args.input_epoch), domain, score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt_dir', type=str, default='ckpts/', help='Directory for parameter checkpoints')
    parser.add_argument('-l', '--load_params', dest='load_params', action='store_true',
                        help='Restore training from previous model checkpoint?')
    parser.add_argument("-o", "--output", type=str, default='cnn', help='Save model filepath')
    parser.add_argument("-i", "--input", type=str, default='haohancnn', help='Load model filepath')
    parser.add_argument("-ie", "--input_epoch", type=int, default=399, help='Load model after n epochs')
    parser.add_argument('-e', '--epochs', type=int, default=1000, help='How many epochs to run in total?')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='Batch size during training per GPU')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed for generating data')
    parser.add_argument('-adv', '--adv_flag', type=int, default=0, help='adversarially training local features')
    parser.add_argument('-m', '--lam', type=float, default=1.0, help='weights of regularization')
    parser.add_argument('-g', '--gpu_id', type=str, default='0', help='gpuid used for trianing')
    parser.add_argument('-row', '--row', type=int, default=0, help='direction delta in row')
    parser.add_argument('-col', '--col', type=int, default=0, help='direction delta in column')
    parser.add_argument('-ng', '--ngray', type=int, default=16, help='regularization gray level')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    # test_cnn(args)
    # modelname = 'Adv_filter1'
    # filenames = os.listdir('/home/songweig/AdvLF/results/Cifar10/longer_pretrain')

    test(args)

    # filenames = os.listdir('/home/songweig/AdvLF/results/Cifar10/longer_pretrain')
    # for filename in filenames:
    #     if filename.endswith('.txt') and 'm1' in filename:
    #         modelname = filename[:-4]
    #         args.input = modelname
    #         print(modelname)
    #         test(args)
