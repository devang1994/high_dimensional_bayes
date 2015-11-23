import theano
from theano import tensor as T
import numpy as np
from load_synthetic import load_synthetic as load


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


def model(X, w_h, w_o, b_h, b_o):
    h = T.nnet.relu(T.dot(X, w_h) + b_h)
    op = T.dot(h, w_o) + b_o
    return op


def run_test(L2reg=10, hidden_width=10, mini_batchsize=100):
    X_train, X_test, y_train, y_test = load()

    X = T.fmatrix(name='X')
    Y = T.fmatrix(name='Y')

    input_size = X_train.shape[1]
    w_h = init_weights((input_size, hidden_width))
    w_o = init_weights((hidden_width, 1))
    b_h = init_bias(hidden_width)
    b_o = init_bias(1)

    op = model(X, w_h, w_o, b_h, b_o)
    params = [w_h, w_o, b_h, b_o]

    cost = (T.mean(T.sqr(op - Y))) + T.sum(w_h ** 2) * L2reg + T.sum(w_o ** 2) * L2reg
    updates = sgd(cost, params)

    train = theano.function(inputs=[X, Y], outputs=cost,
                            updates=updates, allow_input_downcast=True,
                            name='train')
    predict = theano.function(inputs=[X], outputs=op, allow_input_downcast=True)
    fcost = theano.function(inputs=[op, Y], outputs=cost, allow_input_downcast=True)

    for i in range(100):
        for start, end in zip(range(0, len(X_train), mini_batchsize),
                              range(mini_batchsize, len(X_train), mini_batchsize)):
            yd = (floatX(y_train[start:end])).reshape(mini_batchsize, 1)
            cost_v = train(X_train[start:end], yd)

        fin_cost_test = fcost(predict(X_test), floatX(y_test).reshape(len(y_test), 1))
        fin_cost_train = fcost(predict(X_train), floatX(y_train).reshape(len(y_train), 1))
        print i, fin_cost_test, fin_cost_train

    print 'final b_o values'
    print b_o.get_value()

    fin_cost_test = fcost(predict(X_test), floatX(y_test).reshape(len(y_test), 1))
    fin_cost_train = fcost(predict(X_train), floatX(y_train).reshape(len(y_train), 1))
    print 'Hwidth: {}, BatchSize: {}, L2reg: {},Train: {}, Test: {}'.format(hidden_width, mini_batchsize, L2reg,
                                                                             fin_cost_train, fin_cost_test)


if __name__ == "__main__":
    run_test(L2reg=1,hidden_width=10)
