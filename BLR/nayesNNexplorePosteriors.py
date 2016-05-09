from bayesNN_HMCv2 import sampler_on_BayesNN, objective
import numpy as np
from math import sqrt


def mixing():
    ntrain = 20
    noise_var = 0.01
    X_train = np.random.uniform(low=-1.0, high=1.0, size=ntrain).reshape(ntrain, 1)
    # print X_train.shape
    y_train = objective(X_train) + np.random.randn(ntrain, 1) * sqrt(noise_var)

    precisions = [7, 10, 10, 7]
    vy = 10
    hWidths = [50, 50, 50]

    train_err, test_err, samples, train_op_samples = sampler_on_BayesNN(burnin=0, n_samples=1000, precisions=precisions,
                                                                        vy=vy,
                                                                        X_train=X_train, y_train=y_train,
                                                                        hWidths=hWidths)

    print samples.shape


mixing()
