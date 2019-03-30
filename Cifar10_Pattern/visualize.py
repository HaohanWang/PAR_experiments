import os
import matplotlib.pyplot as plt

# file_names = ['_0_0.txt', '_0_1.txt', '_0_2.txt']
file_names = ['_1_2.txt']

for file_name in file_names:
    plt.clf()
    adv_file = os.path.join('performance', 'Adv_0.1'+file_name)
    base_file = os.path.join('performance', 'ResNet'+file_name)
    out_file = os.path.join('plots', file_name[1:4]+'.png')
    base_training_acc = []
    base_test_acc = []
    with open(base_file) as f:
        for line in f:
            if not line.startswith('Epoch'):
                continue
            if 'Prediction' in line:
                base_test_acc.append(float(line.rstrip().split()[-1]))
            else:
                base_training_acc.append(float(line.rstrip().split()[8][:6]))

    adv_training_acc = []
    adv_test_acc = []
    with open(adv_file) as f:
        for line in f:
            if not line.startswith('Epoch'):
                continue
            if 'Prediction' in line:
                adv_test_acc.append(float(line.rstrip().split()[-1]))
            else:
                adv_training_acc.append(float(line.rstrip().split()[8][:6]))

    fig, ax1 = plt.subplots(1, 1)
    train_epoch = [i for i in range(200)]
    test_epoch = [i*5 for i in range(40)]
    ax1.plot(test_epoch, base_test_acc, 'orange', label='Baseline Test')
    ax1.plot(train_epoch, base_training_acc, 'orange', linestyle='dashed', label='Baseline Train')
    ax1.plot(test_epoch[:3], adv_test_acc, 'turquoise', label='Adv Test')
    ax1.plot(train_epoch[:16], adv_training_acc, 'turquoise', linestyle='dashed', label='Adv Train')

    ax1.set_xlabel('number of epochs')
    ax1.set_ylabel('Accuracy on training and vlidation set')
    ax1.grid(True)
    ax1.legend()
    plt.savefig(out_file)