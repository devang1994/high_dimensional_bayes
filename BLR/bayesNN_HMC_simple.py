import numpy

from theano import function, shared
from theano import tensor as T
import theano
from hmc import HMC_sampler
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt

def objective(x):
    return (np.sin(x * 7) + np.cos(x * 17))

def floatX(X):
    """convert to np array with floatX"""
    return np.asarray(X, dtype=theano.config.floatX)


def init_weights(shape):
    """theano shared variable with given shape"""
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def init_bias(n):
    """theano shared variable with given size"""
    return theano.shared(value=np.zeros((n,), dtype=theano.config.floatX))


# TODO for now have fixed batchsize to 1 , maybe change

# TODO ask amar how to sensibly sample variance / uncertainty


# def model(X, w_h1, w_o, b_h1, b_o):
#     h1 = T.tanh(T.dot(X, w_h1) + b_h1)
#
#     op = T.dot(h1, w_o) + b_o
#     return op

theano.config.optimizer='fast_compile'

def unpack_theta(theta,hidden_width,input_size,output_size,index=0):


    w1=theta[index,0:hidden_width*input_size].reshape((input_size,hidden_width))


    b1=theta[index,hidden_width*input_size:hidden_width*input_size+hidden_width]
    wo=theta[index,hidden_width*input_size+hidden_width:hidden_width*input_size+hidden_width+hidden_width*output_size].reshape((hidden_width,-1))
    temp=hidden_width*input_size+hidden_width+hidden_width*output_size
    bo=theta[index,temp:temp+output_size]

    return w1,b1,wo,bo


def model(X,theta,hidden_width,input_size,output_size):

    w1,b1,wo,bo=unpack_theta(theta,hidden_width,input_size,output_size)
    h1 = T.tanh(T.dot(X, w1) + b1)

    op = T.dot(h1, wo) + bo
    return op


def model_np(X,w1,b1,wo,bo):

    # w1,b1,wo,bo=unpack_theta(theta,hidden_width,input_size,output_size)
    h1 = np.tanh(np.dot(X, w1) + b1)

    op = np.dot(h1, wo) + bo
    return op

def uniform_weights(shape):
    scale = sqrt(6. / (shape[1] + shape[0]))
    return theano.shared(floatX(np.random.uniform(low=-scale, high=scale, size=shape)))


def sampler_on_BayesNN(burnin, n_samples,v1,vy,hidden_width,X_train,y_train):
    #simple architecture 1->20-> 1
    #same variance of both layers assumed


    batchsize = 1

    rng = numpy.random.RandomState(123)
    # set hyper parameters for bayes NN


    #  energy function for a multi-variate Gaussian
    def NN_energy(theta):
        # theta is a configuration of the params of the neural network


        # return 0.5 * (theano.tensor.dot((x - mu), cov_inv) *
        #               (x - mu)).sum(axis=1)

        prior_comp= T.sum(T.sqr(theta))* v1 /2.0
        prediction=model(X_train,theta,hidden_width,1,1)
        # print 'shape of ytrain '
        # print y_train.shape
        # print 'shape of pred '
        # print prediction.shape.eval()
        temp=T.sqr(y_train-prediction) #TODO error is here need to fix prediction size
        data_comp=0.5 * vy *T.sum(temp)
                                            # just squaring

        return prior_comp+data_comp

    # Declared shared random variable for positions
    input_size=1
    output_size=1

    dim =hidden_width*(input_size+1)+(hidden_width+1)*output_size
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
    a=sampler.draw()
    print type(a)
    print type(garbage)
    print len(garbage)
    _samples = numpy.asarray([sampler.draw() for r in range(n_samples)])
    # Flatten to [n_samples * batchsize, dim]
    print len(_samples)
    samples = _samples.T.reshape(dim, -1).T
    _samples.T
    print type(samples)
    print samples.shape

    print samples[0,:]
    print samples[1,:]


    def make_predictions_from_NNsamples(X,samples):

        op_samples=[]
        for i in range(len(samples)):
            w1,b1,wo,bo=unpack_theta(input_size=1,output_size=1,hidden_width=hidden_width,theta=samples,index=i)

            op=model_np(X,w1,b1,wo,bo)

            # if(i==1 or i==2):
            #     print 'prinitng about op'
            #     print op.shape
                # print op # seems fine
            # print op.shape
            op_samples.append(op)

        # print op_samples[1]
        print op_samples[1].shape

        op_samples = (np.asarray(op_samples))
        print 'shape after transforming'
        op_samples=op_samples.reshape(n_samples,-1)
        print op_samples.shape

        y_pred = np.average(op_samples,axis=0) # averaged over all the NN's

        y_sd = np.std(op_samples,axis=0)


        print 'shape of ypred {}'.format(y_pred.shape)
        print 'shape of y_sd {}'.format(y_sd.shape)
        y_pred=y_pred.reshape(len(y_pred),1)
        y_sd=y_sd.reshape(len(y_pred),1)
        return op_samples,y_pred,y_sd

    op_samples,y_pred,y_sd=make_predictions_from_NNsamples(X_train,samples)

    print 'printing op samples stuff'
    print op_samples.shape
    print (op_samples[1,:])

    print 'many samples of pred y[2] , shape is {}'.format(op_samples[:,2].shape)
    print op_samples[:,2]


    ntest=1000
    X_test = np.linspace(-1., 2., ntest)
    y_test = objective(X_test)
    X_test=X_test.reshape(ntest,1)
    y_test=y_test.reshape(ntest,1)
    op_samples,y_pred_test,y_sd_test=make_predictions_from_NNsamples(X_test,samples)


    print y_sd_test[0:40]

    print 'train error '
    print MSE(y_train,y_pred)
    print 'test error '
    print y_test.shape
    print y_pred_test.shape
    print MSE (y_test,y_pred_test)

    # print y_pred_test[0:40]

    plt.plot(X_test,y_test,linewidth=2,color='black',label='Objective')
    plt.plot(X_train,y_train,'ro',label='Data')
    plt.plot(X_test,y_pred_test+2*y_sd_test,label='Credible',color='blue')
    plt.plot(X_test,y_pred_test-2*y_sd_test,label='Interval',color='blue')
    plt.plot(X_test,y_pred_test,label='Prediction',color='green')

    # plt.plot(X_train,y_pred)
    plt.legend()

    plt.savefig('logs/BNN_logs/BNNv1{}vy{}hW{}'.format(v1,vy,hidden_width),dpi=300)
    plt.show()
    #

def test_hmc():
    ntrain=100
    noise_var = 0.01
    X_train = np.random.uniform(low=-1.0, high=1.0, size=ntrain).reshape(ntrain,1)
    print X_train.shape
    y_train =  objective(X_train) + np.random.randn(ntrain,1) * sqrt(noise_var)
    print y_train.shape
    sampler = sampler_on_BayesNN(burnin=1000, n_samples=1000,v1=10,vy=500,
                                 hidden_width=50,X_train=X_train,y_train=y_train)

test_hmc()


# xtrain_ = np.random.uniform(low=-1.0, high=1.0, size=ntrain)
# ytrain_ = objective(xtrain_) + np.random.randn(ntrain) * noise
# xtest_ = np.linspace(-1., 2., ntest)
# ytest_ = objective(xtest_)
#
# X_train = xtrain_.reshape(len(xtrain_), 1)
# y_train = ytrain_