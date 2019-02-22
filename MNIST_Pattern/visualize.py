__author__ = 'Haohan Wang'

import numpy as np

from matplotlib import pyplot as plt

import matplotlib

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)

def loadTxt(path_filename):
    TR = []
    VAL = []
    TE = []
    for i in range(5):
        updateTest = True
        maxVal = 0
        text = [line.strip() for line in open(path_filename + '_' + str(i) + '.txt')]
        tr = []
        val = []
        te = []
        for line in text:
            if line.startswith('Epoch'):
                items = line.split()
                tr.append(float(items[8][:-1]))
                val.append(float(items[-1]))
                if len(val) == 0:
                    updateTest = True
                else:
                    if val[-1] > maxVal:
                        updateTest = True
                        maxVal = val[-1]
                    else:
                        updateTest = False
            if line.startswith('Best'):
                if updateTest:
                    te.append(float(line.split()[-1]))
                else:
                    te.append(te[-1])
        print te[-1]
        TR.append(tr)
        VAL.append(val)
        TE.append(te[:-1])
    TR = np.array(TR)
    VAL = np.array(VAL)
    TE = np.array(TE)

    return TR, VAL, TE

def loadTxtNew(path_filename):
    TR = []
    VAL = []
    TE = []
    for i in range(5):
        updateTest = True
        maxVal = 0
        text = [line.strip() for line in open(path_filename+ '_' + str(i) + '.txt')]
        tr = []
        val = []
        te = []
        startUpdate = False
        for line in text:
            if line.startswith('Start'):
                startUpdate = True
            if startUpdate:
                if line.startswith('Epoch'):
                    items = line.split()
                    tr.append(float(items[8][:-1]))
                    val.append(float(items[-1]))
                    if len(val) == 0:
                        updateTest = True
                    else:
                        if val[-1] > maxVal:
                            updateTest = True
                            maxVal = val[-1]
                        else:
                            te.append(te[-1])
                if line.startswith('Best'):
                    if updateTest:
                        te.append(float(line.split()[-1]))

        TR.append(tr)
        VAL.append(val)
        TE.append(te[:-1])
    TR = np.array(TR)
    VAL = np.array(VAL)
    TE = np.array(TE)

    return TR, VAL, TE

def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(range(mean.shape[0]), ub, lb,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(mean, color_mean)

def plot(corr=0):
    tr1, val1, te1 = loadTxt('baseline_'+str(corr))
    tr2, val2, te2 = loadTxt('hex_'+str(corr))

    plot_mean_and_CI(np.mean(tr1, 0), np.mean(tr1, 0)-np.std(tr1,0), np.mean(tr1, 0)+np.std(tr1,0), color_mean='b--', color_shading='c')
    plot_mean_and_CI(np.mean(te1, 0), np.mean(te1, 0)-np.std(te1,0), np.mean(te1, 0)+np.std(te1,0), color_mean='b', color_shading='c')
    plot_mean_and_CI(np.mean(val1, 0), np.mean(val1, 0)-np.std(val1,0), np.mean(val1, 0)+np.std(val1,0), color_mean='b.', color_shading='c')

    plot_mean_and_CI(np.mean(tr2, 0), np.mean(tr2, 0)-np.std(tr2,0), np.mean(tr2, 0)+np.std(tr2,0), color_mean='r--', color_shading='m')
    plot_mean_and_CI(np.mean(te2, 0), np.mean(te2, 0)-np.std(te2,0), np.mean(te2, 0)+np.std(te2,0), color_mean='r', color_shading='m')
    plot_mean_and_CI(np.mean(val2, 0), np.mean(val2, 0)-np.std(val2,0), np.mean(val2, 0)+np.std(val2,0), color_mean='r.', color_shading='m')

    plt.legend(loc=4)
    plt.ylim(0.4, 1.05)
    plt.savefig('MNIST_Pattern_Confound_'+str(corr)+'.pdf')
    plt.clf()

def resultPlot():
    boxColors = ['darkkhaki', 'royalblue']

    fig = plt.figure(dpi=350, figsize=(25, 5))
    axs = [0 for i in range(10)]

    newFiles = ['pre', 'info', 'local', 'local1']

    fileNames = ['baseline',  'hex', 'pre', 'info', 'local1', 'local']
    labelNames = ['B', 'H', 'D', 'I', 'G1', 'G2']

    plt.style.use('bmh')

    for i in range(6):
        axs[i] = fig.add_axes([0.075+i*0.15, 0.1, 0.12, 0.7])

        ts = []
        if i < 3:
            for k in range(len(fileNames)):
                if fileNames[k] in newFiles:
                    tr, val, te = loadTxtNew('../results/MNIST_Pattern/'+ fileNames[k]+'_'+str(i))
                else:
                    tr, val, te = loadTxt('../results/MNIST_Pattern/'+ fileNames[k]+'_'+str(i))
                ts.append(te[:,-1])
        else:
            for k in range(len(fileNames)):
                if fileNames[k] in newFiles:
                    tr, val, te = loadTxtNew('../results/MNIST_Pattern_Confound/'+ fileNames[k]+'_'+str(i%3))
                else:
                    tr, val, te = loadTxt('../results/MNIST_Pattern_Confound/'+ fileNames[k]+'_'+str(i%3))
                ts.append(te[:,-1])

        # m1 = np.mean(r1)
        # s1 = np.std(r1)
        # m2 = np.mean(r2)
        # s2 = np.std(r2)

        # axs[c].errorbar(x=[0, 1], y=[m1, m2], yerr=[s1, s2])

        axs[i].boxplot(ts, positions=[j for j in range(len(fileNames))], widths=[0.5 for j in range(len(fileNames))])
        # axs[c].boxplot(r2, positions=[1])

        axs[i].set_xlim(-0.5, len(fileNames)-0.5)
        axs[i].set_ylim(0, 1.1)

        if i == 0:
            axs[i].set_ylabel('Accuracy')
        axs[i].set_xticklabels(labelNames)
        # if c1 == 0:
        # axs[c].set_xticks([0, 1], ['NN', 'HEX-NN'])
        # else:
        #     axs[c].get_xaxis().set_visible(False)
        if i == 0:
            axs[i].title.set_text('original\nindependent')
        elif i == 1:
            axs[i].title.set_text('random\nindependent')
        elif i == 2:
            axs[i].title.set_text('radial\nindependent')
        elif i == 3:
            axs[i].title.set_text('original\ndependent')
        elif i == 4:
            axs[i].title.set_text('random\ndependent')
        elif i == 5:
            axs[i].title.set_text('radial\ndependent')


    # plt.legend(loc="upper center", bbox_to_anchor=(1, 1), fancybox=True, ncol=2)
    plt.savefig('fig.pdf', dpi=350, format='pdf')

if __name__ == '__main__':
    # for i in range(3):
    #     plot(i)
    resultPlot()
