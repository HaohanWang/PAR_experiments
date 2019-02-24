f = open('mnist_r.sh', 'w')
for iter in range(10):
    for corr in range(6):
        f.write('python cnn.py -adv 1 -m 0.01 -test %s -g 1 -e 1000 > results/ALF_%s_0.01_%s.txt'%(str(corr), str(corr), str(iter))+'\n\n')
