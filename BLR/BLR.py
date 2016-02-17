__author__ = 'da368'
import numpy as np
import matplotlib.pyplot as plt
from theano import tensor as T
import theano
from math import sqrt
from theano.misc.pkl_utils import StripPickler

from sklearn.metrics import mean_squared_error as MSE


epochs = 4000


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


def model(X, w_h1, w_h2, w_h3, w_o, b_h1, b_h2, b_h3, b_o):
    h1 = T.tanh(T.dot(X, w_h1) + b_h1)
    h2 = T.tanh(T.dot(h1, w_h2) + b_h2)
    h3 = T.tanh(T.dot(h2, w_h3) + b_h3)
    op = T.dot(h3, w_o) + b_o
    return op


def model_act(X, w_h1, w_h2, w_h3, w_o, b_h1, b_h2, b_h3, b_o):
    # returns activations for third hidden layer
    h1 = T.tanh(T.dot(X, w_h1) + b_h1)
    h2 = T.tanh(T.dot(h1, w_h2) + b_h2)
    h3 = T.tanh(T.dot(h2, w_h3) + b_h3)
    op = T.dot(h3, w_o) + b_o
    return h3


def uniform_weights(shape):
    scale = sqrt(6. / (shape[1] + shape[0]))
    return theano.shared(floatX(np.random.uniform(low=-scale, high=scale, size=shape)))


def BLR(alpha, beta, phi_train, phi_test, y_train):
    # each column of phi is a data point

    nbasis = phi_train.shape[1]

    # TODO confirm alpha or alhpa^2
    K = np.add(np.multiply(alpha * alpha, np.eye(nbasis)),
               np.multiply(beta, np.dot(np.transpose(phi_train), phi_train)))

    Kphi_test = np.dot(np.linalg.inv(K), np.transpose(phi_test))

    # at all the test values

    mu = np.dot(np.transpose(Kphi_test), np.dot(np.transpose(phi_train), y_train)) * beta

    s2 = np.sum(np.multiply(np.transpose(phi_test), Kphi_test), 0) + (1.0 / beta)
    s2 = s2.reshape(len(s2), 1)

    return mu, s2


def objective(x):
    return (np.sin(x * 7) + np.cos(x * 17))


def exp1():
    # simple BLR test run

    noise = 0.2
    ntrain = 50
    ntest = 1000
    nperbatch = 20


    # -- generate training data and fix test inputs
    # -- the trailing underscore indicates that these are flattened arrays
    xtrain_ = np.random.uniform(low=0.0, high=1.0, size=ntrain)
    ytrain_ = objective(xtrain_) + np.random.randn(ntrain) * noise
    xtest_ = np.linspace(-1., 2., ntest)

    phi_train = xtrain_.reshape(len(xtrain_), 1)
    y_train = ytrain_
    phi_test = xtest_.reshape(len(xtest_), 1)
    print phi_train.shape

    mu, s2 = BLR(0.2, 4, phi_train, phi_test, y_train)

    print s2.shape, mu.shape
    s = np.sqrt(s2) * 2
    mu = mu.reshape(len(mu), 1)  # temp measure remove
    plt.plot(xtest_, mu, color='r', label='posterior')
    plt.plot(xtest_, mu - s, color='blue', label='credible')
    plt.plot(xtest_, mu + s, color='blue', label='interval')
    plt.plot(xtest_, objective(xtest_), color='black')
    plt.plot(xtrain_, ytrain_, 'ro')
    #
    # plt.title('Bayesian linear regression with learned features')
    plt.legend()
    plt.savefig('abc.png')
    plt.show()


def MSE_reg(Y, op, params, L2reg):
    #defines the cost function
    cost = (T.mean(T.sqr(op - Y)))
    # penalizing all params
    for i in params:
        cost = cost + T.sum(i ** 2) * L2reg

    return cost


def mlp_synthetic(X_train, X_test, y_train, y_test, L2reg=0.01, hidden_width=50, mini_batchsize=5):
    X = T.fmatrix(name='X')
    Y = T.fmatrix(name='Y')

    input_size = X_train.shape[1]
    print input_size
    w_h1 = uniform_weights((input_size, hidden_width))
    w_h2 = uniform_weights((hidden_width, hidden_width))
    w_h3 = uniform_weights((hidden_width, hidden_width))
    b_h1 = init_bias(hidden_width)
    b_h2 = init_bias(hidden_width)
    b_h3 = init_bias(hidden_width)

    w_o = uniform_weights((hidden_width, 1))
    #b_h = init_bias(hidden_width)
    b_o = init_bias(1)

    op = model(X, w_h1, w_h2, w_h3, w_o, b_h1, b_h2, b_h3, b_o)
    params = [w_h1, w_h2, w_h3, w_o, b_h1, b_h2, b_h3, b_o]
    cost = MSE_reg(Y, op, params, L2reg=L2reg)
    updates = sgd(cost, params)
    # updates=Adam(cost,params)
    train = theano.function(inputs=[X, Y], outputs=cost,
                            updates=updates, allow_input_downcast=True,
                            name='train')
    predict = theano.function(inputs=[X], outputs=op, allow_input_downcast=True)
    fcost = theano.function(inputs=[op, Y], outputs=cost, allow_input_downcast=True)

    test_costs = []
    train_costs = []

    for i in range(epochs):
        for start, end in zip(range(0, len(X_train), mini_batchsize),
                              range(mini_batchsize, len(X_train), mini_batchsize)):
            yd = (floatX(y_train[start:end])).reshape(mini_batchsize, 1)
            # print (X_train[start:end]).shape
            cost_v = train(X_train[start:end], yd)

        # Done this cost prediction needs to change
        # fin_cost_test = fcost(predict(X_test), floatX(y_test).reshape(len(y_test), 1))
        # fin_cost_train = fcost(predict(X_train), floatX(y_train).reshape(len(y_train), 1))
        fin_cost_test = MSE(predict(X_test), y_test)
        fin_cost_train = MSE(predict(X_train), y_train)
        test_costs.append(fin_cost_test)
        train_costs.append(fin_cost_train)
        # print i, fin_cost_test, fin_cost_train

    # print 'final b_o values'
    # print b_o.get_value()

    # fin_cost_test = fcost(predict(X_test), floatX(y_test).reshape(len(y_test), 1))
    # fin_cost_train = fcost(predict(X_train), floatX(y_train).reshape(len(y_train), 1))
    fin_cost_test = MSE(predict(X_test), y_test)
    fin_cost_train = MSE(predict(X_train), y_train)
    # print 'NumTP: {}, Hwidth: {}, BatchSize: {}, L2reg: {}, Seed {},Train: {}, Test: {}'.format(numTrainPoints,
    #                                                                                             hidden_width,
    #                                                                                             mini_batchsize, L2reg,
    #                                                                                             rand_seed,
    #                                                                                             fin_cost_train,
    #                                                                                             fin_cost_test)


    # Calculate RMS error with simple mean prediction
    test_mean = np.mean(y_test)
    train_mean = np.mean(y_train)

    mean_p_test = np.ones(y_test.size) * test_mean
    mean_p_train = np.ones(y_train.size) * train_mean

    # test_cost=fcost(floatX(mean_p_test).reshape(len(y_test), 1), floatX(y_test).reshape(len(y_test), 1))
    # train_cost=fcost(floatX(mean_p_train).reshape(len(y_train), 1), floatX(y_train).reshape(len(y_train), 1))
    test_cost = MSE(mean_p_test, y_test)
    train_cost = MSE(mean_p_train, y_train)

    tArray = np.ones(epochs) * test_cost
    # print 'MSE for mean prediction, Train:{} ,Test:{}'.format(train_cost,test_cost)
    ref_err=MSE_reference(y_test)

    # ref_arr=ref_err*np.ones(epochs)
    # plt.plot(range(epochs), test_costs, label='Test')
    # plt.plot(range(epochs),train_costs,label='Train')
    # plt.plot(range(epochs),ref_arr,label='Ref')
    # plt.xlabel('Epochs')
    # plt.ylabel('Error')
    # plt.title('TrainCost:{}, TestCost: {}'.format(fin_cost_train, fin_cost_test))
    # plt.legend()
    # plt.show()
    # plt.close()


    dest_pkl = 'my_test.pkl'
    f = open(dest_pkl, 'wb')
    strip_pickler = StripPickler(f, protocol=-1)
    strip_pickler.dump(params)
    f.close()

    h3 = model_act(X, w_h1, w_h2, w_h3, w_o, b_h1, b_h2, b_h3, b_o)
    transform = theano.function(inputs=[X], outputs=h3, allow_input_downcast=True)

    test_transformed = transform(X_test)
    train_transformed = transform(X_train)
    test_predictions = predict(X_test)
    # returns the transformed test data, ie the activations from the third hidden layer
    return fin_cost_train, fin_cost_test, train_transformed, test_transformed , test_predictions

def MSE_reference(y_test):


    mean_pred=np.mean(y_test)
    mean_arr=np.ones(y_test.shape)*mean_pred

    return MSE(y_test,mean_arr)

def prob_model():

    noise = 0.2
    ntrain = 50
    ntest = 1000
    nperbatch = 20


    # -- generate training data and fix test inputs
    # -- the trailing underscore indicates that these are flattened arrays
    xtrain_ = np.random.uniform(low=0.0, high=1.0, size=ntrain)
    ytrain_ = objective(xtrain_) + np.random.randn(ntrain) * noise
    xtest_ = np.linspace(-1., 2., ntest)
    ytest_ = objective(xtest_)

    X_train = xtrain_.reshape(len(xtrain_), 1)
    y_train = ytrain_
    y_test = ytest_
    X_test = xtest_.reshape(len(xtest_), 1)

    ref_err=MSE_reference(y_test)
    print 'ref err: {}'.format(ref_err)

    fin_cost_train, fin_cost_test, train_transformed, test_transformed, test_predictions = mlp_synthetic(X_train, X_test, y_train, y_test,
                                                                                       L2reg=0.0, hidden_width=50,
                                                                                       mini_batchsize=5)
    print 'train {}, test {}'.format(fin_cost_train,fin_cost_test)

    phi_train = train_transformed
    y_train = ytrain_
    phi_test = test_transformed
    print phi_train.shape

    mu, s2 = BLR(0.2, 4, phi_train, phi_test, y_train)

    print s2.shape, mu.shape
    s = np.sqrt(s2) * 2
    mu = mu.reshape(len(mu), 1)  # temp measure remove


    # plt.plot(xtest_, mu, color='r', label='posterior')
    # plt.plot(xtest_, mu - s, color='blue', label='credible')
    # plt.plot(xtest_, mu + s, color='blue', label='interval')
    # plt.plot(xtest_, objective(xtest_), color='black')
    # plt.plot(xtrain_, ytrain_, 'ro')
    # #
    # # plt.title('Bayesian linear regression with learned features')
    # plt.legend()
    # plt.savefig('abc.png')
    # plt.show()

def diag_test():

    noise = 0.2
    ntrain = 50
    ntest = 1000
    nperbatch = 20


    # -- generate training data and fix test inputs
    # -- the trailing underscore indicates that these are flattened arrays
    xtrain_ = np.random.uniform(low=0.0, high=1.0, size=ntrain)
    ytrain_ = objective(xtrain_) + np.random.randn(ntrain) * noise
    xtest_ = np.linspace(-1., 2., ntest)
    ytest_ = objective(xtest_)

    X_train = xtrain_.reshape(len(xtrain_), 1)
    y_train = ytrain_
    y_test = ytest_
    X_test = xtest_.reshape(len(xtest_), 1)


    ref_err=MSE_reference(y_test)
    print 'ref err: {}'.format(ref_err)

    plt.plot(X_test,y_test, label = 'Objective')
    plt.plot(X_train,y_train,'ro', label='Train Data')


    fin_cost_train, fin_cost_test, train_transformed, test_transformed, test_predictions = mlp_synthetic(X_train, X_test, y_train, y_test,
                                                                                       L2reg=0.0, hidden_width=50,
                                                                                       mini_batchsize=5)
    plt.plot(X_test,test_predictions,label='Predictions')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    #prob_model()
    diag_test()