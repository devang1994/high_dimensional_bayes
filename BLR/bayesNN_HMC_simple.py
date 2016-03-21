import numpy

from theano import function, shared
from theano import tensor as T
import theano
from hmc import HMC_sampler


def model(X, w_h1, w_o, b_h1, b_o):
    h1 = T.tanh(T.dot(X, w_h1) + b_h1)

    op = T.dot(h1, w_o) + b_o
    return op


def sampler_on_BayesNN(sampler_cls, burnin, n_samples,model,v1,v2,vy):
    batchsize = 3

    rng = numpy.random.RandomState(123)
    # set hyper parameters for bayes NN





    #  energy function for a multi-variate Gaussian
    def NN_energy(x):

        return 0.5 * (theano.tensor.dot((x - mu), cov_inv) *
                      (x - mu)).sum(axis=1)

    # Declared shared random variable for positions
    position = rng.randn(batchsize, dim).astype(theano.config.floatX)
    position = theano.shared(position)

    # Create HMC sampler
    sampler = sampler_cls(position, gaussian_energy,
                          initial_stepsize=1e-3, stepsize_max=0.5)