from bayesNN_HMCv2 import sampler_on_BayesNN, objective, analyse_samples
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import cPickle as pickle


def mixing(sf, vy, show_fit=False):
    '''

    :param sf: scale factor for precisions
    :param vy: precision of noise
    :return:
    '''
    ntrain = 20
    noise_var = 0.01
    X_train = np.random.uniform(low=-1.0, high=1.0, size=ntrain).reshape(ntrain, 1)
    # print X_train.shape
    y_train = objective(X_train) + np.random.randn(ntrain, 1) * sqrt(noise_var)

    # precisions = [0.6125875773048164, 0.03713439386866191, 14.22759780450891, 5.72501724650353]
    # vy = 4.631095917555727

    precisions = [1, 1]

    precisions = [sf * x for x in precisions]

    hWidths = [100]

    train_err, test_err, samples, train_op_samples = sampler_on_BayesNN(burnin=0, n_samples=1000, precisions=precisions,
                                                                        vy=vy,
                                                                        X_train=X_train, y_train=y_train,
                                                                        hWidths=hWidths, target_acceptance_rate=0.9)

    w1 = samples[:, 1]
    w2 = samples[:, 3]
    w3 = samples[:, 9]
    # w4 = samples[:, 200]

    plt.figure()
    plt.plot(w1, label='w1')
    plt.plot(w2, label='w2')
    plt.plot(w3, label='w3')
    plt.title('weight prec {}, noise prec {}'.format(sf, vy))
    plt.legend()

    plt.xlabel('Num Iterations')
    plt.ylabel('Value')

    # plt.savefig('logs/BNN_logs/mixingWeightsPrec10L', dpi=300)

    print samples.shape

    analyse_samples(samples, X_train, y_train, hWidths=hWidths, burnin=200, display=True)

    samples = samples[200:, :]  # burning in

    w1 = samples[:, 1]
    w2 = samples[:, 5200]
    w3 = samples[:, 1200]
    plt.figure()

    N = samples.shape[0]
    n = N / 10

    plt.hist(w1, bins=n)  # bin it into n = N/10 bins
    plt.figure()

    plt.hist(w2, bins=n)  # bin it into n = N/10 bins
    plt.figure()

    plt.hist(w3, bins=n)  # bin it into n = N/10 bins


if __name__ == '__main__':
    # mixing_from_pickle()
    # sfVals=[1,10]
    # vVals=[1,10]
    # for sf in sfVals:
    #     for vy in vVals:
    #         mixing(sf,vy)
    #
    # mixing(1,1,show_fit=True)
    # mixing(1,10,show_fit=True)
    mixing(1, 100, show_fit=True)
    # mixing(10,1,show_fit=True)
    plt.show()
