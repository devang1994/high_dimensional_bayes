import numpy as np
import matplotlib.pyplot as pl
from theano import tensor as T
import theano
from math import sqrt
from sklearn import gaussian_process


import GPy
from theano.misc.pkl_utils import StripPickler
def floatX(X):
    """convert to np array with floatX"""
    return np.asarray(X, dtype=theano.config.floatX)


def init_weights(shape):
    """theano shared variable with given shape"""
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))


def init_bias(n):
    """theano shared variable with given size"""
    return theano.shared(value=np.zeros((n,), dtype=theano.config.floatX))


def sgd(cost, params, lr=0.005):
    """same as simple update, just does it over all params"""
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates


def model(X, w_h1,w_h2,w_h3, w_o, b_h1,b_h2,b_h3, b_o):
    h1 = T.nnet.tanh(T.dot(X, w_h1) + b_h1)
    h2 = T.nnet.tanh(T.dot(X, w_h2) + b_h2)
    h3 = T.nnet.tanh(T.dot(X, w_h3) + b_h3)
    op = T.dot(h3, w_o) + b_o
    return op


def uniform_weights(shape):
    scale = sqrt(6. / (shape[1] + shape[0]))
    return theano.shared(floatX(np.random.uniform(low=-scale, high=scale, size=shape)))



def BLR(alpha,beta, phi_train,phi_test,y_train):
    # each col is a data point

    nbasis=phi_train.shape[1]
    # K = torch.addmm(alpha^2, torch.eye(nbasis), beta, phi_train:t(), phi_train)


    #TODO confirm alpha or alpha or alhpa^2
    K=np.add(np.multiply(alpha*alpha,np.eye(nbasis)), np.multiply(beta,np.dot(np.transpose(phi_train),phi_train)))
    #basically Sn-1

    Kphi_test = np.dot(np.linalg.inv(K), np.transpose(phi_test))



    # at all the test values

    mu = np.dot ( np.transpose(Kphi_test), np.dot(np.transpose(phi_train),y_train)) * beta

    #s2 = torch.cmul(phi:t(), Kphi):sum(1):add(1/beta)
    s2= np.sum(np.multiply(np.transpose(phi_test), Kphi_test),0) + (1.0/beta)
    s2=s2.reshape(len(s2),1)

    return mu,s2


def objective(x):
    return (np.sin(x*7) + np.cos(x*17))

def exp1():
    #simple BLR test run

    noise = 0.2
    ntrain = 50
    ntest = 1000
    nperbatch = 20


    # -- generate training data and fix test inputs
    # -- the trailing underscore indicates that these are flattened arrays
    xtrain_ = np.random.uniform(low=0.0,high=1.0,size=ntrain)
    ytrain_ = objective(xtrain_) + np.random.randn(ntrain)*noise
    xtest_ = np.linspace(-1., 2., ntest)

    phi_train=xtrain_.reshape(len(xtrain_),1)
    y_train=ytrain_
    phi_test=xtest_.reshape(len(xtest_),1)
    print phi_train.shape

    mu, s2= BLR(0.2, 4,phi_train,phi_test,y_train)

    print s2.shape,mu.shape
    s=np.sqrt(s2)*2
    mu=mu.reshape(len(mu),1) # temp measure remove
    plt.plot(xtest_, mu, color='r',label='posterior')
    plt.plot(xtest_, mu-s, color='blue',label='credible')
    plt.plot(xtest_, mu+s, color='blue',label='interval')
    plt.plot(xtest_,objective(xtest_),color='black')
    plt.plot(xtrain_,ytrain_,'ro')
    #
    # plt.title('Bayesian linear regression with learned features')
    plt.legend()
    plt.savefig('abc.png')
    plt.show()


def play_gpy(ntrain=10):


    xtrain_ = np.random.uniform(low=-1.0, high=2.0, size=ntrain)
    X_train = xtrain_.reshape(len(xtrain_), 1)

    y_train = objective(X_train)

    x_test = np.linspace(-1, 2, 1000)
    gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
    gp.fit(X_train, y_train)
    y_pred, sigma2_pred = gp.predict(x_test, eval_MSE=True)
    sigma = np.sqrt(sigma2_pred)

    fig = pl.figure()
    pl.plot(x_test, f(x_test), 'r:', label=u'$f(x) = x\,\sin(x)$')
    pl.plot(X_train, y_train, 'r.', markersize=10, label=u'Observations')
    pl.plot(x_test, y_pred, 'b-', label=u'Prediction')
    pl.fill(np.concatenate([x_test, x_test[::-1]]),
            np.concatenate([y_pred - 1.9600 * sigma,
                           (y_pred + 1.9600 * sigma)[::-1]]),
            alpha=.5, fc='b', ec='None', label='95% confidence interval')
    pl.xlabel('$x$')
    pl.ylabel('$f(x)$')
    pl.ylim(-10, 20)
    pl.legend(loc='upper left')
    pl.show()
def f(x):
    return x * np.sin(x)

def objective(x):
    return (np.sin(x * 7) + np.cos(x * 17))



if __name__== '__main__':

    play_gpy()

    # X = T.fmatrix(name='X')
    #
    # a= np.ones((2,2))
    # print a
    # w_h1 = uniform_weights((2,2))
    # k=T.dot(w_h1,X)
    # predict = theano.function(inputs=[X], outputs=k, allow_input_downcast=True)
    #
    # print predict(a)
    # dest_pkl = 'my_test.pkl'
    # f = open(dest_pkl, 'wb')
    # strip_pickler = StripPickler(f, protocol=-1)
    # strip_pickler.dump(w_h1)
    # f.close()

