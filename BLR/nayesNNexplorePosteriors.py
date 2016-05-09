from bayesNN_HMCv2 import sampler_on_BayesNN, objective
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


def mixing():
    ntrain = 20
    noise_var = 0.01
    X_train = np.random.uniform(low=-1.0, high=1.0, size=ntrain).reshape(ntrain, 1)
    # print X_train.shape
    y_train = objective(X_train) + np.random.randn(ntrain, 1) * sqrt(noise_var)

    precisions = [1, 1, 1, 1]
    vy = 10
    hWidths = [50, 50, 50]

    train_err, test_err, samples, train_op_samples = sampler_on_BayesNN(burnin=0, n_samples=2000, precisions=precisions,
                                                                        vy=vy,
                                                                        X_train=X_train, y_train=y_train,
                                                                        hWidths=hWidths, target_acceptance_rate=0.6)

    w1 = samples[:, 1]
    w2 = samples[:, 5200]
    w3 = samples[:, 1200]
    w4 = samples[:, 200]

    plt.plot(w1)
    plt.plot(w2)
    plt.plot(w3)
    # plt.plot(w4)

    plt.xlabel('Num Iterations')
    plt.ylabel('Value')

    plt.savefig('logs/BNN_logs/mixingWeightsPrec10L', dpi=300)

    print samples.shape

    samples = samples[200:, :]  # burning in

    w1 = samples[:, 1]
    w2 = samples[:, 5200]
    w3 = samples[:, 1200]
    plt.figure()

    N = samples.shape[0]
    n = N / 10

    p, x = np.histogram(w1, bins=n)  # bin it into n = N/10 bins
    x = x[:-1] + (x[1] - x[0]) / 2  # convert bin edges to centers
    f = UnivariateSpline(x, p, s=n)
    plt.plot(x, f(x))
    plt.figure()

    p, x = np.histogram(w2, bins=n)  # bin it into n = N/10 bins
    x = x[:-1] + (x[1] - x[0]) / 2  # convert bin edges to centers
    f = UnivariateSpline(x, p, s=n)
    plt.plot(x, f(x))
    plt.figure()

    p, x = np.histogram(w3, bins=n)  # bin it into n = N/10 bins
    x = x[:-1] + (x[1] - x[0]) / 2  # convert bin edges to centers
    f = UnivariateSpline(x, p, s=n)
    plt.plot(x, f(x))

    plt.show()

mixing()
