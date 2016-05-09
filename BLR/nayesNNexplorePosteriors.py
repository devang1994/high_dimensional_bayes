from bayesNN_HMCv2 import sampler_on_BayesNN, objective
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt


def mixing():
    ntrain = 20
    noise_var = 0.01
    X_train = np.random.uniform(low=-1.0, high=1.0, size=ntrain).reshape(ntrain, 1)
    # print X_train.shape
    y_train = objective(X_train) + np.random.randn(ntrain, 1) * sqrt(noise_var)

    precisions = [10, 10, 10, 10]
    vy = 10
    hWidths = [50, 50, 50]

    train_err, test_err, samples, train_op_samples = sampler_on_BayesNN(burnin=0, n_samples=5000, precisions=precisions,
                                                                        vy=vy,
                                                                        X_train=X_train, y_train=y_train,
                                                                        hWidths=hWidths)

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

    plt.figure()

    plt.hist(w1, bins=100, histtype='step', normed=True)
    plt.figure()
    plt.hist(w2, bins=100, histtype='step', normed=True)
    plt.figure()
    plt.hist(w3, bins=100, histtype='step', normed=True)

    plt.show()

mixing()
