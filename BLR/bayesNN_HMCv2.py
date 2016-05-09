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


def acquisition_UCB(m, s, k=0.2):
    a = m - k * s
    return a


# TODONE for now have fixed batchsize to 1 , maybe change
# TODO think about other input_oput sizes
# TODONE ask amar how to sensibly sample variance / uncertainty

theano.config.optimizer = 'fast_compile'


# theano.config.device = 'cpu'

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
# TODONE posterior isnt moving
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
    scales_prior = scales[:]
    shapes_prior = shapes[:]

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
    num_sampled_log = [num_sampled]
    # TODONE study evolution of errors
    while num_sampled < n_samples:
        pass
        # first update params of gamma
        # then draw
        # TODONE use this to debug, samples shape (10, 5251), train_op_samples (10, 100)
        # TODONE Updates of gamma params is wrong debug


        # find weights and biases of last sample
        weights, biases = unpack_theta(samples, hWidths, input_size, output_size, index=(samples.shape[0] - 1))

        for i in range(len(weights)):
            a = np.sum(np.square(weights[i])) + np.sum(np.square(biases[i]))

            b = weights[i].size + biases[i].size

            shapes[i] = shapes_prior[i] + b / 2.0
            scales[i] = 1.0 / (1.0 / scales_prior[i] + 0.5 * a)
            # scales[i] = 2.0 / ((2.0 / scales[i]) + a)
            # shapes[i] = shapes[i] + b / 2.0

            gamma_samples[i] = np.random.gamma(shapes[i], scales[i])

            # print 'a {} , b {}'.format(a, b)
            # print '-------'

            # precisions on weights

        b = len(train_op_samples[(train_op_samples.shape[0] - 1), :])  # length of train set
        a = np.sum(np.square(last_train_op_sampled - y_train))

        i = len(hWidths) + 1
        scales[i] = 1.0 / (1.0 / scales_prior[i] + 0.5 * a)
        shapes[i] = shapes_prior[i] + b / 2.0
        gamma_samples[i] = np.random.gamma(shapes[i], scales[i])

        # can use the previous positions of theta to seed this sampler
        train_err, test_err, samples, train_op_samples = sampler_on_BayesNN(burnin=20, n_samples=20,
                                                                            precisions=gamma_samples[
                                                                                       0:(len(hWidths) + 1)],
                                                                            vy=gamma_samples[len(hWidths) + 1],
                                                                            X_train=X_train, y_train=y_train,
                                                                            hWidths=hWidths)

        train_errs.append(train_err)
        test_errs.append(test_err)

        if (num_sampled % 60 == 0):
            print 'num_sampled {}'.format(num_sampled)
            print 'scales {}'.format(scales)
            print 'shapes {}'.format(shapes)
            print 'gamma samples{}'.format(gamma_samples)
            print 'train err {}, test err {}'.format(train_err, test_err)
            print '-------------------------------'

        last_train_op_sampled = train_op_samples[(train_op_samples.shape[0] - 1), :].reshape(train_op_samples.shape[1],
                                                                                             1)

        fin_samples = np.vstack((fin_samples, samples))
        num_sampled += 20
        num_sampled_log.append(num_sampled)
        # if num_sampled%30==0:
        #     print num_sampled
        #     print 'scales {}, shapes {}'.format(scales,shapes)
        #     print 'precisions {}'.format(gamma_samples)
        #     print 'train err {}, test err {}'.format(train_err,test_err)

    return fin_samples, train_errs, test_errs, num_sampled_log


# TODONE edit to accept initial position
# TODONE edit to output actual samples
# TODONE make func to do combined Gibbs sampling

# TODONE when doing Gibbs which sample to initialize next HMC from

def sampler_on_BayesNN(burnin, n_samples, precisions, vy, hWidths, X_train, y_train, init_theta=None,
                       target_acceptance_rate=0.9):
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
                                                    initial_stepsize=1e-3, stepsize_max=0.5,
                                                    target_acceptance_rate=target_acceptance_rate)

    # Start with a burn-in process
    # print 'about to sample'
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
    #  this is what would need to change for o/put sizes other than one
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
    #
    # ntest = 1000
    # X_test = np.linspace(-1., 1., ntest)
    # y_test = objective(X_test)
    # X_test = X_test.reshape(ntest, 1)
    # y_test = y_test.reshape(ntest, 1)
    # op_samples, y_pred_test, y_sd_test = make_predictions_from_NNsamples(X_test, samples)

    # print y_sd_test[0:40]

    # print 'train error '

    train_err = MSE(y_train, y_pred)
    train_err = -1

    # print 'test error '
    # print y_test.shape
    # print y_pred_test.shape
    # test_err = MSE(y_test, y_pred_test)

    test_err = -1
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

def analyse_samples(samples, X_train, y_train, hWidths, burnin=0, display=False):
    '''

    :param samples:
    :param X_train:
    :param y_train:
    :param hWidths:
    :param burnin: discards the first few samples, makes sure the we start accepting after a few gibbs
     sampling steps
    :return:
    '''

    samples = samples[burnin:]
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]

    n_samples = len(samples)

    def make_predictions_from_NNsamples(X, samples, y):
        op_samples = []
        errs = []
        for i in range(len(samples)):
            weights, biases = unpack_theta(samples, hWidths, input_size, output_size, index=i)

            op = model_np(X, weights, biases)

            op_samples.append(op)

            err = MSE(op, y)

            errs.append(err)

        op_samples = (np.asarray(op_samples))

        op_samples = op_samples.reshape(n_samples, -1)

        y_pred = np.average(op_samples, axis=0)  # averaged over all the NN's

        y_sd = np.std(op_samples, axis=0)

        y_pred = y_pred.reshape(len(y_pred), 1)
        y_sd = y_sd.reshape(len(y_pred), 1)
        return op_samples, errs, y_pred, y_sd

    train_preds, train_errs, train_pred, train_sd = make_predictions_from_NNsamples(X_train, samples, y_train)
    ntest = 1000
    X_test = np.linspace(-1., 2., ntest)
    y_test = objective(X_test)
    X_test = X_test.reshape(ntest, 1)
    y_test = y_test.reshape(ntest, 1)
    test_preds, test_errs, test_pred, test_sd = make_predictions_from_NNsamples(X_test, samples, y_test)
    # print train_errs

    # print len(train_errs)

    if (display):
        plt.figure(1)
        sample_plot(X_train, y_train, X_test, y_test, test_pred, test_sd)

        plt.figure(2)
        plt.plot(train_errs, label='train')
        plt.plot(test_errs, label='test')
        plt.legend()
        plt.show()

    return test_pred, test_sd

def sample_plot(X_train, y_train, X_test, y_test, y_pred_test, y_sd_test):
    plt.plot(X_test, y_test, linewidth=2, color='black', label='Objective')
    plt.plot(X_train, y_train, 'ro', label='Data')
    plt.plot(X_test, y_pred_test + 2 * y_sd_test, label='Credible', color='blue')
    plt.plot(X_test, y_pred_test - 2 * y_sd_test, label='Interval', color='blue')
    plt.plot(X_test, y_pred_test, label='Prediction', color='green')

    # plt.plot(X_train,y_pred)
    plt.legend()

    # plt.savefig('logs/BNN_logs/BNNv2{}vy{}hW{}.png'.format(precisions, vy, hWidths), dpi=300)

    # plt.clf()
    # plt.show()
    #
    # print 'samples shape {}, train_op_samples {}'.format(samples.shape,train_op_samples.shape)
    # samples shape (10, 5251), train_op_samples (10, 100)

def test_combinedGibbs():
    ntrain = 20
    noise_var = 0.01
    X_train = np.random.uniform(low=-1.0, high=1.0, size=ntrain).reshape(ntrain, 1)
    # print X_train.shape
    y_train = objective(X_train) + np.random.randn(ntrain, 1) * sqrt(noise_var)
    scales = [2., 2., 2., 2., 0.5]
    c = 0.125 * 0.5
    scales = [c * x for x in scales]
    shapes = [5., 5., 5., 5., 20.]
    shapes = [x / c for x in shapes]
    f_samples, train_errs, test_errs, numSampledLog = combinedGibbsHMC_BayesNN(100, [50, 50, 50], X_train, y_train,
                                                                               scales=scales, shapes=shapes)

    # scales and shapes chosen to have a normal like distribution with mean around 10
    analyse_samples(f_samples, X_train, y_train, hWidths=[50, 50, 50], burnin=50)

def produce_mu_and_sd(n_samples, hWidths, xtrain, ytrain, scales, shapes, burnin=0):
    f_samples, train_errs, test_errs, numSampledLog = combinedGibbsHMC_BayesNN(100, [50, 50, 50], xtrain, ytrain,
                                                                               scales=scales, shapes=shapes)

    test_pred, test_sd = analyse_samples(f_samples, xtrain, ytrain, hWidths=[50, 50, 50], burnin=burnin)

    return test_pred, test_sd

def bayes_opt(func, initial_random=2, k=0.2, num_it=20):
    '''function to do bayesOpt on and number of initial random evals
    noise is artificially added to objective function calls when training
    '''
    noise = 0.2
    ntest = 1000
    ntrain = initial_random  # number of initial random function evals
    xtrain = np.random.uniform(low=-1.0, high=1.0, size=(ntrain, 1))
    print xtrain.shape
    ytrain = func(xtrain) + np.random.randn(ntrain, 1) * noise
    print ytrain.shape
    xtest = np.linspace(-1., 2., ntest)
    xtest = xtest.reshape(ntest, 1)
    ytest = func(xtest)
    plt.figure(1)  # the first figure

    plt.plot(xtest, func(xtest), color='black')
    plt.plot(xtrain, ytrain, 'ro')

    for i in range(num_it):
        print 'it:{}'.format(i)

        scales = [2., 2., 2., 2., 0.5]
        c = 0.125 * 0.5
        scales = [c * x for x in scales]
        shapes = [5., 5., 5., 5., 20.]
        shapes = [x / c for x in shapes]

        mu, sd = produce_mu_and_sd(100, [50, 50, 50], xtrain, ytrain, scales=scales, shapes=shapes, burnin=40)

        alpha = acquisition_UCB(mu, sd, k=k)

        index = np.argmin(alpha)
        next_query = xtest[index]
        next_y = func(next_query) + np.random.randn(1, 1) * noise

        s = sd  # standard deviations


        # plt.plot(xtest, objective(xtest), color='black',label='objective')
        # plt.plot(xtrain, ytrain, 'ro')
        # plt.plot(xtest, mu, color='r', label='posterior')
        # plt.plot(xtest, mu - s, color='blue', label='credible')
        # plt.plot(xtest, mu + s, color='blue', label='interval')
        # plt.plot(xtest,alpha,label='acquistion func',color='green')
        # plt.title('BLR with learned features ,ntrain:{}'.format(xtrain.shape[0]))
        # plt.legend()
        # plt.savefig('bayesOptNtrain{}k{}init{}.png'.format(xtrain.shape[0],k,ntrain),dpi=300)
        # xtrain=np.vstack((xtrain,next_query))
        # ytrain=np.vstack((ytrain,next_y))

        if (i % 2 == 0):
            plt.figure()
            f, axarr = plt.subplots(2, sharex=True)

            # .scatter(x, y)
            axarr[1].plot(xtest, func(xtest), color='black', label='objective', linewidth=2.0)
            axarr[1].plot(xtrain, ytrain, 'ro')
            axarr[1].plot(xtest, mu, color='r', label='posterior')
            # plt.plot(xtest, mu - s, color='blue', label='credible')
            # plt.plot(xtest, mu + s, color='blue', label='interval')
            axarr[1].plot(xtest, alpha, label='acquistion func', color='green')
            # plt.plot(xtest,np.zeros(ntest),color='black')
            axarr[0].plot(xtest, s, label='sigma', color='blue')
            axarr[0].set_title('BLR with learned features ,ntrain:{}'.format(xtrain.shape[0]))
            plt.legend(fontsize='x-small')
            # plt.savefig('bayesOptNtrain{}k{}init{}.png'.format(xtrain.shape[0], k, ntrain), dpi=300)

        xtrain = np.vstack((xtrain, next_query))
        ytrain = np.vstack((ytrain, next_y))
    plt.show()




if __name__ == '__main__':
    # test_hmc()

    # test_combinedGibbs()
    print theano.config.device

    func = objective

    bayes_opt(func, initial_random=10, num_it=6, k=5)
