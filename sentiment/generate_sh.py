f = open('sentiment.sh', 'w')
for iter in range(5):
for corr in range(10):
    f.write('python cnn.py -adv 1 -m 0.01 -test %s -g 0 -e 1000 > results/ALF_%s_0.01_%s.txt'%(str(corr), str(corr), str(iter))+'\n\n')
