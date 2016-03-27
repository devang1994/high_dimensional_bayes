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

# TODO ask amar how to sensibly sample variance / uncertainty

theano.config.optimizer = 'fast_compile'


def unpack_theta(theta, hWidths , input_size, output_size, index=0):
    # w1 = theta[index, 0:hidden_width * input_size].reshape((input_size, hidden_width))
    # b1 = theta[index, hidden_width * input_size:hidden_width * input_size + hidden_width]
    # wo = theta[index,
    #      hidden_width * input_size + hidden_width:hidden_width * input_size + hidden_width + hidden_width * output_size].reshape(
    #     (hidden_width, -1))
    # temp = hidden_width * input_size + hidden_width + hidden_width * output_size
    # bo = theta[index, temp:temp + output_size]
    #
    # return w1, b1, wo, bo
    weights=[]
    biases=[]
    widths=hWidths[:]
    widths.insert(0,input_size)
    widths.append(output_size)
    # print widths
    cur=0

    for i in range(len(widths)-1):
        w=theta[index,cur:cur+widths[i]*widths[i+1]].reshape((widths[i], widths[i+1]))
        cur=cur+widths[i]*widths[i+1]
        weights.append(w)
        b=theta[index,cur : cur+widths[i+1]]
        biases.append(b)
        cur= cur+widths[i+1]


    return weights,biases

# w,b=unpack_theta(np.arange(5701).reshape(1,5701),[50,50,50],1,1)
# # print w
# print len(b)
#
# print w[0]
# print w[1]
# print w[2]
# print w[3]

def model(X, theta, hWidths, input_size, output_size):
    weights,biases = unpack_theta(theta, hWidths, input_size, output_size)

    h=X
    for i in range(len(weights)-1):
        w=weights[i]
        b=biases[i]
        h=T.tanh(T.dot(h,w)+b)

    op=T.dot(h,weights[-1])+biases[-1]
    return op


def model_np(X, weights,biases):
    # w1,b1,wo,bo=unpack_theta(theta,hidden_width,input_size,output_size)

    h=X
    for i in range(len(weights)-1):
        w=weights[i]
        b=biases[i]
        h=np.tanh(np.dot(h,w)+b)

    op=np.dot(h,weights[-1])+biases[-1]
    return op


def find_dim_theta(hWidths,input_size,output_size):

    dim =np.sum(hWidths)+output_size # bisaes

    for i in range(len(hWidths)-1):
        dim = dim +hWidths[i]*hWidths[i+1]

    dim=dim +hWidths[0]*input_size
    dim=dim +hWidths[-1]*output_size
    return dim
    pass


# TODO edit to accept initial position
# TODO edit to output actual samples
# TODO make func to do combined Gibbs sampling

def sampler_on_BayesNN(burnin, n_samples, precisions, vy, hWidths, X_train, y_train):
    """

    Test dataset is just linspace(-1,1,1000)

    :param burnin: samples to burnin
    :param n_samples: num samples
    :param precisions: [v1,v2,v3,v4]
    :param vy: variance of output
    :param hWidths: [h1,h2,h3]
    :param X_train: train data
    :param y_train: test data
    :type vy: float
    :return: test and train error


    """



    batchsize = 1

    input_size=X_train.shape[1]
    output_size=y_train.shape[1]

    rng = numpy.random.RandomState(123)
    # set hyper parameters for bayes NN


    #  energy function for a multi-variate Gaussian
    def NN_energy(theta):
        # theta is a configuration of the params of the neural network


        # return 0.5 * (theano.tensor.dot((x - mu), cov_inv) *
        #               (x - mu)).sum(axis=1)

        # prior_comp = T.sum(T.sqr(theta)) * v1 / 2.0
        weights,biases=unpack_theta(theta,hWidths,input_size,output_size)

        prior_comp=(T.sum(T.sqr(weights[0]))+ T.sum(T.sqr(biases[0])))* precisions[0] /2.0
        # print len(weights)
        # print len(precisions)
        for i in range(1,len(weights)):
            # temp=T.append(T.ravel(weights[i]),biases[i])
            # print temp.shape
            prior_comp=prior_comp+(T.sum(T.sqr(weights[i]))+ T.sum(T.sqr(biases[i])))* precisions[i] /2.0


        prediction = model(X_train, theta, hWidths,input_size,output_size)
        # print 'shape of ytrain '
        # print y_train.shape
        # print 'shape of pred '
        # print prediction.shape.eval()
        temp = T.sqr(y_train - prediction)  # TODO error is here need to fix prediction size
        data_comp = 0.5 * vy * T.sum(temp)
        # just squaring

        return prior_comp + data_comp

    # Declared shared random variable for positions
    input_size = 1
    output_size = 1

    dim = find_dim_theta(hWidths,input_size,output_size)
    position = rng.randn(batchsize, dim).astype(theano.config.floatX)
    position = theano.shared(position)

    # Create HMC sampler
    sampler = HMC_sampler.new_from_shared_positions(position, NN_energy,
                                                    initial_stepsize=1e-3, stepsize_max=0.5)


    # Start with a burn-in process
    print 'about to sample'
    garbage = [sampler.draw() for r in range(burnin)]  # burn-in Draw
    # `n_samples`: result is a 3D tensor of dim [n_samples, batchsize,
    # dim]
    a = sampler.draw()
    # print type(a)
    # print type(garbage)
    # print len(garbage)
    _samples = numpy.asarray([sampler.draw() for r in range(n_samples)])
    # Flatten to [n_samples * batchsize, dim]
    # print len(_samples)
    samples = _samples.T.reshape(dim, -1).T
    # _samples.T
    # print type(samples)
    # print samples.shape

    # print samples[0, :]
    # print samples[1, :]

    def make_predictions_from_NNsamples(X, samples):
        op_samples = []
        for i in range(len(samples)):
            weights,biases = unpack_theta(samples,hWidths,input_size,output_size,index=i)

            op = model_np(X, weights,biases)

            # if(i==1 or i==2):
            #     print 'prinitng about op'
            #     print op.shape
            # print op # seems fine
            # print op.shape
            op_samples.append(op)

        # print op_samples[1]
        # print op_samples[1].shape

        op_samples = (np.asarray(op_samples))
        print 'shape after transforming'
        op_samples = op_samples.reshape(n_samples, -1)
        # print op_samples.shape
        y_pred = np.average(op_samples, axis=0)  # averaged over all the NN's

        y_sd = np.std(op_samples, axis=0)

        # print 'shape of ypred {}'.format(y_pred.shape)
        # print 'shape of y_sd {}'.format(y_sd.shape)
        y_pred = y_pred.reshape(len(y_pred), 1)
        y_sd = y_sd.reshape(len(y_pred), 1)


        return op_samples, y_pred, y_sd

    op_samples, y_pred, y_sd = make_predictions_from_NNsamples(X_train, samples)

    # print 'printing op samples stuff'
    # print op_samples.shape
    # print (op_samples[1, :])

    # print 'many samples of pred y[2] , shape is {}'.format(op_samples[:, 2].shape)
    # print op_samples[:, 2]

    ntest = 1000
    X_test = np.linspace(-1., 1., ntest)
    y_test = objective(X_test)
    X_test = X_test.reshape(ntest, 1)
    y_test = y_test.reshape(ntest, 1)
    op_samples, y_pred_test, y_sd_test = make_predictions_from_NNsamples(X_test, samples)

    # print y_sd_test[0:40]

    print 'train error '
    train_err = MSE(y_train, y_pred)
    print 'test error '
    # print y_test.shape
    # print y_pred_test.shape
    test_err= MSE(y_test, y_pred_test)

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


    return train_err,test_err

    #


def test_hmc():
    ntrain = 100
    noise_var = 0.01
    X_train = np.random.uniform(low=-1.0, high=1.0, size=ntrain).reshape(ntrain, 1)
    # print X_train.shape
    y_train = objective(X_train) + np.random.randn(ntrain, 1) * sqrt(noise_var)
    # print y_train.shape



    trains={}
    tests={}
    vyVals=[1,10,30,70,100,150,300]
    cvals=[0.01,0.1,1,10,100]
    for c in cvals:


        precisions=[7,10,10,7]
        precisions = [c*x for x in precisions]
        # scaling precisions. contant base val found by sqrt(nin+nout)
        plt.figure()
        temp=[]
        temp1=[]

        for vy in vyVals:
            train_err,test_err = sampler_on_BayesNN(burnin=1000, n_samples=1000, precisions=precisions,
                                 vy=vy,X_train=X_train, y_train=y_train,hWidths=[50,50,50])
            print precisions, vy
            print train_err,test_err
            temp.append(train_err)
            temp1.append(test_err)
            trains[(c,vy)]=train_err
            tests[(c,vy)]=test_err



        plt.plot(vyVals,temp,label='Train')
        plt.plot(vyVals,temp1,label='Test')
        plt.semilogx()
        plt.semilogy()
        plt.legend()
        plt.title('c={}'.format(c))

    v = 100

    temp = []
    temp1=[]
    plt.figure()
    for c in cvals:
        train_err=trains[(c,v)]
        test_err=tests[(c,v)]
        temp.append(train_err)
        temp1.append(test_err)

    plt.plot(cvals, temp, label='Train')
    plt.plot(cvals, temp1, label='Test')
    plt.semilogx()
    plt.semilogy()
    plt.legend()
    plt.title('v={}'.format(v))


    pickle.dump(tests, open( "logs/BNN_logs/tests.p", "wb" ))
    pickle.dump(trains, open( "logs/BNN_logs/trains.p", "wb" ))
    plt.show()



if __name__=='__main__':

    test_hmc()
