import os
import matplotlib.pyplot as plt

file_names = ['pretrain_adv_conv0_0.1', 'pretrain_adv_conv0_1', 'pretrain_adv_conv1_0.1', 'pretrain_adv_conv1_1']

for file_name in file_names:
    plt.clf()
    file = os.path.join('performance', file_name+'.txt')
    out_file = os.path.join('plots', file_name+'.png')
    normal = []
    greyscale = []
    negative = []
    with open(file) as f:
        for line in f:
            if line.startswith('Starting'):
                continue
            accuracy = float(line.rstrip().split('')[-1])
            if 'greyscale' in line:
                greyscale.append(accuracy)
            elif 'negative' in line:
                negative.append(accuracy)
            else:
                normal.append(accuracy)

    baseline_n = [0.9169 for i in range(10)]
    baseline_g = [0.8575 for i in range(10)]
    baseline_ne = [0.6088 for i in range(10)]

    fig, ax1 = plt.subplots(1, 1)
    n_epoch = [i*50 for i in range(10)]
    ax1.plot(n_epoch, normal, 'orange', label='Normal Acc')
    ax1.plot(n_epoch, baseline_n, 'orange', linestyle='dashed')
    ax1.plot(n_epoch, greyscale, 'turquoise', label='Greyscale Acc')
    ax1.plot(n_epoch, baseline_g, 'turquoise', linestyle='dashed')
    ax1.plot(n_epoch, negative, 'firebrick', label='Negative Acc')
    ax1.plot(n_epoch, baseline_ne, 'firebrick', linestyle='dashed')

    ax1.set_xlabel('number of epochs')
    ax1.set_ylabel('loss on training and vlidation set')
    ax1.grid(True)
    ax1.legend()
    plt.savefig(out_file)