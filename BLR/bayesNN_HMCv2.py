__author__ = 'da368'
import numpy

from theano import function, shared
from theano import tensor as T
import theano
from hmc import HMC_sampler
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt

np.random.seed(42)
import cPickle as pickle


def objective(x):
    return (np.sin(x * 7) + np.cos(x * 17))


# TODO for now have fixed batchsize to 1 , maybe change
# TODO think about other input_oput sizes
# TODO ask amar how to sensibly sample variance / uncertainty

theano.config.optimizer = 'fast_compile'


def unpack_theta(theta, hWidths, input_size, output_size, index=0):
    # w1 = theta[index, 0:hidden_width * input_size].reshape((input_size, hidden_width))
    # b1 = theta[index, hidden_width * input_size:hidden_width * input_size + hidden_width]
    # wo = theta[index,
    #      hidden_width * input_size + hidden_width:hidden_width * input_size + hidden_width + hidden_width * output_size].reshape(
    #     (hidden_width, -1))
    # temp = hidden_width * input_size + hidden_width + hidden_width * output_size
    # bo = theta[index, temp:temp + output_size]
    #
    # return w1, b1, wo, bo
    weights = []
    biases = []
    widths = hWidths[:]
    widths.insert(0, input_size)
    widths.append(output_size)
    # print widths
    cur = 0

    for i in range(len(widths) - 1):
        w = theta[index, cur:cur + widths[i] * widths[i + 1]].reshape((widths[i], widths[i + 1]))
        cur = cur + widths[i] * widths[i + 1]
        weights.append(w)
        b = theta[index, cur: cur + widths[i + 1]]
        biases.append(b)
        cur = cur + widths[i + 1]

    return weights, biases


def model(X, theta, hWidths, input_size, output_size):
    weights, biases = unpack_theta(theta, hWidths, input_size, output_size)

    h = X
    for i in range(len(weights) - 1):
        w = weights[i]
        b = biases[i]
        h = T.tanh(T.dot(h, w) + b)

    op = T.dot(h, weights[-1]) + biases[-1]
    return op


def model_np(X, weights, biases):
    # w1,b1,wo,bo=unpack_theta(theta,hidden_width,input_size,output_size)

    h = X
    for i in range(len(weights) - 1):
        w = weights[i]
        b = biases[i]
        h = np.tanh(np.dot(h, w) + b)

    op = np.dot(h, weights[-1]) + biases[-1]
    return op


def find_dim_theta(hWidths, input_size, output_size):
    dim = np.sum(hWidths) + output_size  # bisaes

    for i in range(len(hWidths) - 1):
        dim = dim + hWidths[i] * hWidths[i + 1]

    dim = dim + hWidths[0] * input_size
    dim = dim + hWidths[-1] * output_size
    return dim
    pass


# TODO burnin WRT to gibbs
# TODO posterior isnt moving
def combinedGibbsHMC_BayesNN(n_samples, hWidths, X_train, y_train, scales, shapes):
    """

    :param n_samples:
    :param precisions:
    :param vy:
    :param hWidths:
    :param X_train:
    :param y_train:
    :param scales: params for gibbs . len is one more than precisions to sample vy as well
    :param shapes: params for gibbs
    :return:
    """

    input_size = X_train.shape[1]
    output_size = y_train.shape[1]

    # pick precisions etc from prior
    gamma_samples = []
    for i in range(len(scales)):
        gamma_samples.append(np.random.gamma(shapes[i], scales[i]))

    print 'prior gamma_samples {}'.format(gamma_samples)

    train_err, test_err, samples, train_op_samples = sampler_on_BayesNN(burnin=10, n_samples=10,
                                                                        precisions=gamma_samples[0:(len(hWidths) + 1)],
                                                                        vy=gamma_samples[len(hWidths) + 1],
                                                                        X_train=X_train, y_train=y_train,
                                                                        hWidths=hWidths)  # initial samples with big burnin

    num_sampled = 10
    fin_samples = samples

    #  gibbs sampling
    last_train_op_sampled = train_op_samples[(train_op_samples.shape[0] - 1), :].reshape(train_op_samples.shape[1], 1)
    train_errs = [train_err]
    test_errs = [test_err]
    # TODO study evolution of errors
    while num_sampled < n_samples:
        pass
        # first update params of gamma
        # then draw
        # TODO use this to debug, samples shape (10, 5251), train_op_samples (10, 100)
        # TODO Updates of gamma params is wrong debug


        # find weights and biases of last sample
        weights, biases = unpack_theta(samples, hWidths, input_size, output_size, index=(samples.shape[0] - 1))

        for i in range(len(weights)):
            a = np.sum(np.square(weights[i])) + np.sum(np.square(biases[i]))

            b = weights[i].size + biases[i].size
            scales[i] = 2.0 / ((2.0 / scales[i]) + a)  # TODO possibly wrong
            shapes[i] = shapes[i] + b / 2.0  # TODO  possibly wrong

            gamma_samples[i] = np.random.gamma(shapes[i], scales[i])

            print 'a {} , b {}'.format(a, b)
            print 'scales {}'.format(scales)
            print 'shapes {}'.format(shapes)
            print 'gamma {}'.format(gamma_samples)

            # precisions on weights

        b = len(train_op_samples[(train_op_samples.shape[0] - 1), :])  # length of train set
        a = np.sum(np.square(last_train_op_sampled - y_train))

        i = len(hWidths) + 1
        scales[i] = 2.0 / ((2.0 / scales[i]) + a)
        shapes[i] = shapes[i] + b / 2.0
        gamma_samples[i] = np.random.gamma(shapes[i], scales[i])

        train_err, test_err, samples, train_op_samples = sampler_on_BayesNN(burnin=10, n_samples=10,
                                                                            precisions=gamma_samples[
                                                                                       0:(len(hWidths) + 1)],
                                                                            vy=gamma_samples[len(hWidths) + 1],
                                                                            X_train=X_train, y_train=y_train,
                                                                            hWidths=hWidths)

        train_errs.append(train_err)
        test_errs.append(test_err)

        last_train_op_sampled = train_op_samples[(train_op_samples.shape[0] - 1), :].reshape(train_op_samples.shape[1],
                                                                                             1)

        fin_samples = np.vstack((fin_samples, samples))
        num_sampled += 10
        # if num_sampled%30==0:
        #     print num_sampled
        #     print 'scales {}, shapes {}'.format(scales,shapes)
        #     print 'precisions {}'.format(gamma_samples)
        #     print 'train err {}, test err {}'.format(train_err,test_err)

    return fin_samples, train_errs, test_errs


# TODONE edit to accept initial position
# TODO edit to output actual samples
# TODO make func to do combined Gibbs sampling

# TODO when doing Gibbs which sample to initialize next HMC from

def sampler_on_BayesNN(burnin, n_samples, precisions, vy, hWidths, X_train, y_train, init_theta=None):
    """

    Test dataset is just linspace(-1,1,1000)

    :param burnin: samples to burnin
    :param n_samples: num samples
    :param precisions: [v1,v2,v3,v4]
    :param vy: precision of output
    :param hWidths: [h1,h2,h3]
    :param X_train: train data
    :param y_train: test data
    :param init_theta: (batchszie,dim) array
    :type vy: float
    :return: test and train error and samples

     shape ofsamples is (nsamples,dim) where dim is the dimension
    train_op_samples has the vals of f(x) for all the returned samples for all

    """

    batchsize = 1

    input_size = X_train.shape[1]
    output_size = y_train.shape[1]

    rng = numpy.random.RandomState(123)
    # set hyper parameters for bayes NN
    #  energy function for a multi-variate Gaussian
    def NN_energy(theta):
        # theta is a configuration of the params of the neural network

        weights, biases = unpack_theta(theta, hWidths, input_size, output_size)

        prior_comp = (T.sum(T.sqr(weights[0])) + T.sum(T.sqr(biases[0]))) * precisions[0] / 2.0
        for i in range(1, len(weights)):
            prior_comp = prior_comp + (T.sum(T.sqr(weights[i])) + T.sum(T.sqr(biases[i]))) * precisions[i] / 2.0

        prediction = model(X_train, theta, hWidths, input_size, output_size)
        temp = T.sqr(y_train - prediction)
        data_comp = 0.5 * vy * T.sum(temp)

        return prior_comp + data_comp

    # Declared shared random variable for positions
    input_size = 1
    output_size = 1

    dim = find_dim_theta(hWidths, input_size, output_size)
    if init_theta is None:
        position = rng.randn(batchsize, dim).astype(theano.config.floatX)
    else:
        position = init_theta.astype(theano.config.floatX)
    position = theano.shared(position)

    # Create HMC sampler
    sampler = HMC_sampler.new_from_shared_positions(position, NN_energy,
                                                    initial_stepsize=1e-3, stepsize_max=0.5)

    # Start with a burn-in process
    print 'about to sample'
    garbage = [sampler.draw() for r in range(burnin)]  # burn-in Draw
    # `n_samples`: result is a 3D tensor of dim [n_samples, batchsize,
    # dim]
    _samples = numpy.asarray([sampler.draw() for r in range(n_samples)])
    # Flatten to [n_samples * batchsize, dim]

    samples = _samples.T.reshape(dim, -1).T  # nsamples,dim
    # _samples.T
    # print type(samples)
    # print 'sample shape {}'.format(samples.shape)

    # print samples[0, :]
    # print samples[1, :]
    # TODO this is what would need to change for o/put sizes other than one
    def make_predictions_from_NNsamples(X, samples):
        op_samples = []
        for i in range(len(samples)):
            weights, biases = unpack_theta(samples, hWidths, input_size, output_size, index=i)

            op = model_np(X, weights, biases)

            op_samples.append(op)

        op_samples = (np.asarray(op_samples))
        op_samples = op_samples.reshape(n_samples, -1)
        y_pred = np.average(op_samples, axis=0)  # averaged over all the NN's

        y_sd = np.std(op_samples, axis=0)

        y_pred = y_pred.reshape(len(y_pred), 1)
        y_sd = y_sd.reshape(len(y_pred), 1)

        return op_samples, y_pred, y_sd

    op_samples, y_pred, y_sd = make_predictions_from_NNsamples(X_train, samples)
    train_op_samples = op_samples  # shape is n_samples, num_test_pts

    ntest = 1000
    X_test = np.linspace(-1., 1., ntest)
    y_test = objective(X_test)
    X_test = X_test.reshape(ntest, 1)
    y_test = y_test.reshape(ntest, 1)
    op_samples, y_pred_test, y_sd_test = make_predictions_from_NNsamples(X_test, samples)

    # print y_sd_test[0:40]

    # print 'train error '
    train_err = MSE(y_train, y_pred)
    # print 'test error '
    # print y_test.shape
    # print y_pred_test.shape
    test_err = MSE(y_test, y_pred_test)

    # print y_pred_test[0:40]

    # plt.plot(X_test, y_test, linewidth=2, color='black', label='Objective')
    # plt.plot(X_train, y_train, 'ro', label='Data')
    # plt.plot(X_test, y_pred_test + 2 * y_sd_test, label='Credible', color='blue')
    # plt.plot(X_test, y_pred_test - 2 * y_sd_test, label='Interval', color='blue')
    # plt.plot(X_test, y_pred_test, label='Prediction', color='green')
    #
    # # plt.plot(X_train,y_pred)
    # plt.legend()
    #
    # plt.savefig('logs/BNN_logs/BNNv2{}vy{}hW{}.png'.format(precisions, vy, hWidths), dpi=300)
    #
    # plt.clf()
    # plt.show()

    # print 'samples shape {}, train_op_samples {}'.format(samples.shape,train_op_samples.shape)
    # samples shape (10, 5251), train_op_samples (10, 100)
    return train_err, test_err, samples, train_op_samples


def test_hmc():
    ntrain = 100
    noise_var = 0.01
    X_train = np.random.uniform(low=-1.0, high=1.0, size=ntrain).reshape(ntrain, 1)
    # print X_train.shape
    y_train = objective(X_train) + np.random.randn(ntrain, 1) * sqrt(noise_var)
    # print y_train.shape



    trains = {}
    tests = {}
    vyVals = [1, 10, 30, 70, 100, 150, 300]
    cvals = [0.01, 0.1, 1, 10, 100]
    for c in cvals:

        precisions = [7, 10, 10, 7]
        precisions = [c * x for x in precisions]
        # scaling precisions. contant base val found by sqrt(nin+nout)
        plt.figure()
        temp = []
        temp1 = []

        for vy in vyVals:
            train_err, test_err, samples, train_op_samp = sampler_on_BayesNN(burnin=1000, n_samples=1000,
                                                                             precisions=precisions,
                                                                             vy=vy, X_train=X_train, y_train=y_train,
                                                                             hWidths=[50, 50, 50])
            print precisions, vy
            print train_err, test_err
            temp.append(train_err)
            temp1.append(test_err)
            trains[(c, vy)] = train_err
            tests[(c, vy)] = test_err

        plt.plot(vyVals, temp, label='Train')
        plt.plot(vyVals, temp1, label='Test')
        plt.semilogx()
        plt.semilogy()
        plt.legend()
        plt.title('c={}'.format(c))

    v = 100

    temp = []
    temp1 = []
    plt.figure()
    for c in cvals:
        train_err = trains[(c, v)]
        test_err = tests[(c, v)]
        temp.append(train_err)
        temp1.append(test_err)

    plt.plot(cvals, temp, label='Train')
    plt.plot(cvals, temp1, label='Test')
    plt.semilogx()
    plt.semilogy()
    plt.legend()
    plt.title('v={}'.format(v))

    pickle.dump(tests, open("logs/BNN_logs/tests.p", "wb"))
    pickle.dump(trains, open("logs/BNN_logs/trains.p", "wb"))
    plt.show()


def test_combinedGibbs():
    ntrain = 100
    noise_var = 0.01
    X_train = np.random.uniform(low=-1.0, high=1.0, size=ntrain).reshape(ntrain, 1)
    # print X_train.shape
    y_train = objective(X_train) + np.random.randn(ntrain, 1) * sqrt(noise_var)

    f_samples, train_errs, test_errs = combinedGibbsHMC_BayesNN(500, [50, 50, 50], X_train, y_train,
                                                                scales=[2, 2, 2, 2, 2], shapes=[5, 5, 5, 5, 5])
    # scales and shapes chosen to have a normal like distribution with mean around 10

    plt.plot(train_errs, label='train')
    plt.plot(test_errs, label='test')
    plt.legend()
    plt.show()
    # TODO plot train and test errors


if __name__ == '__main__':
    # test_hmc()

    test_combinedGibbs()
