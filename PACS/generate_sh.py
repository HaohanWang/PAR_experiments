f = open('PACS.sh', 'w')
lambds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
for iter in range(3):
    for lam in lambds:
        f.write('python alex_run.py -adv 1 -m %s -test 0 -g 0 -e 100 > results/ALF_0_%s_%s.txt'%(str(lam), str(lam), str(iter))+'\n\n')

