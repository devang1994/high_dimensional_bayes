import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import cPickle as pickle


def mixing_from_input(samples):
    plt.figure()
    burnin = 100
    samples = pickle.load(open("logs/BNN_logs/samples_gibbs_acc0.6_2000_sh10.p", "rb"))
    w1 = samples[:, 1]
    w2 = samples[:, 5200]
    w3 = samples[:, 1200]

    plt.plot(w1, label='w1')
    plt.plot(w2, label='w2')
    plt.plot(w3, label='w3')
    plt.legend()
    # plt.plot(w4)

    plt.xlabel('Num Iterations')
    plt.ylabel('Value')

    # plt.savefig('logs/BNN_logs/mixingWeightsPrec10L', dpi=300)

    print samples.shape

    samples = samples[burnin:, :]  # burning in

    w1 = samples[:, 1]
    w2 = samples[:, 5200]
    w3 = samples[:, 1200]
    plt.figure()

    N = samples.shape[0]
    n = N / 10

    plt.hist(w1, bins=n)  # bin it into n = N/10 bins
    # x = x[:-1] + (x[1] - x[0]) / 2  # convert bin edges to centers
    # f = UnivariateSpline(x, p, s=n)
    # plt.plot(x, f(x))
    plt.figure()

    plt.hist(w2, bins=n)  # bin it into n = N/10 bins
    # x = x[:-1] + (x[1] - x[0]) / 2  # convert bin edges to centers
    # f = UnivariateSpline(x, p, s=n)
    # plt.plot(x, f(x))
    plt.figure()

    plt.hist(w3, bins=n)  # bin it into n = N/10 bins
    # x = x[:-1] + (x[1] - x[0]) / 2  # convert bin edges to centers
    # f = UnivariateSpline(x, p, s=n)
    # plt.plot(x, f(x))

    plt.show()
