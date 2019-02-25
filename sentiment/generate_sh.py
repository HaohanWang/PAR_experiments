f = open('sentiment.sh', 'w')
f.write('python cnn.py -adv 1 -m 0.01 -test 9 -g 1 -e 1000 > results/ALF_9_0.01_2.txt'+'\n\n')
for iter in range(3, 5):
    for corr in range(10):
        f.write('python cnn.py -adv 1 -m 0.01 -test %s -g 1 -e 1000 > results/ALF_%s_0.01_%s.txt'%(str(corr), str(corr), str(iter))+'\n\n')
