__author__ = 'da368'
from multiprocessing import Pool
import numpy as np
import bayesNN_HMCv2
from  bayesNN_HMCv2 import objective
from math import sqrt
import cPickle as pickle

def runSampler(kwargs):
    return bayesNN_HMCv2.sampler_on_BayesNN(kwargs['burnin'], kwargs['n_samples'], kwargs['precisions'], kwargs['vy'],
                                            kwargs['hWidths'], kwargs['X_train'], kwargs['y_train'])

def runSampler1(args):
    return bayesNN_HMCv2.sampler_on_BayesNN(*args)


if __name__ == '__main__':
    p = Pool(28)

    listOfArgs = []

    ntrain = 100
    noise_var = 0.01
    X_train = np.random.uniform(low=-1.0, high=1.0, size=ntrain).reshape(ntrain, 1)
    # print X_train.shape
    y_train = objective(X_train) + np.random.randn(ntrain, 1) * sqrt(noise_var)
    # print y_train.shape



    trains = {}
    tests = {}
    vyVals=[1,10,30,70,100,150,300]
    cvals=[0.01,0.1,1,10,100]
    arguments = []

    for c in cvals:

        precisions = [7, 10, 10, 7]
        precisions = [c * x for x in precisions]
        # scaling precisions. contant base val found by sqrt(nin+nout)

        # vyVals=[1,10,30,70,100,150,300]
        for vy in vyVals:
            dict = {'burnin': 1000, 'n_samples': 1000, 'precisions': precisions,
                    'vy': vy, 'X_train': X_train, 'y_train': y_train, 'hWidths': [50, 50, 50]}
            arguments.append((1000, 1000, precisions, vy, [50,50,50],X_train, y_train ))
            listOfArgs.append(dict)

    a = p.map(runSampler1, arguments)

    print a

    pickle.dump(a, open( "logs/BNN_logs/fromPar.p", "wb" ))