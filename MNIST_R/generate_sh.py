# f = open('mnist_r.sh', 'w')
# for iter in range(10):
#     for corr in range(6):
#         f.write('python cnn.py -adv 1 -m 0.01 -test %s -g 1 -e 1000 > results/ALF_%s_0.01_%s.txt'%(str(corr), str(corr), str(iter))+'\n\n')

lambdas = [0.02, 0.05, 0.1, 0.2, 0.5, 1]
corrs = [0, 5]
f = open('mnist_r.sh', 'w')
for iter in range(5):
    for corr in corrs:
        for lam in lambdas:
            f.write('python cnn.py -adv 1 -m %s -test %s -g 1 -e 1000 > results/ALF_%s_%s_%s.txt'%(str(lam), str(corr), str(corr), str(lam), str(iter))+'\n\n')
