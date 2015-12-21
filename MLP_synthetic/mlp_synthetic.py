import theano
from theano import tensor as T
import numpy as np
from load_synthetic import load_synthetic as load
from math import sqrt
from adam import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as MSE

#this is the main code base

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


def uniform_weights(shape):
    scale = sqrt(6. / (shape[1] + shape[0]))
    return theano.shared(floatX(np.random.uniform(low=-scale, high=scale, size=shape)))

def run_test(L2reg=0.01, hidden_width=10, mini_batchsize=100,numTrainPoints=2000):
    X_train, X_test, y_train, y_test = load()

    X_train=X_train[:numTrainPoints]
    y_train=y_train[:numTrainPoints]



    X = T.fmatrix(name='X')
    Y = T.fmatrix(name='Y')

    input_size = X_train.shape[1]
    w_h = uniform_weights((input_size, hidden_width))
    w_o = uniform_weights((hidden_width, 1))
    b_h = init_bias(hidden_width)
    b_o = init_bias(1)

    op = model(X, w_h, w_o, b_h, b_o)
    params = [w_h, w_o, b_h, b_o]

    cost = (T.mean(T.sqr(op - Y))) + T.sum(w_h ** 2) * L2reg + T.sum(w_o ** 2) * L2reg
    updates = sgd(cost, params)
    #updates=Adam(cost,params)
    train = theano.function(inputs=[X, Y], outputs=cost,
                            updates=updates, allow_input_downcast=True,
                            name='train')
    predict = theano.function(inputs=[X], outputs=op, allow_input_downcast=True)
    fcost = theano.function(inputs=[op, Y], outputs=cost, allow_input_downcast=True)

    test_costs=[]
    train_costs=[]
    epochs=1000
    for i in range(epochs):
        for start, end in zip(range(0, len(X_train), mini_batchsize),
                              range(mini_batchsize, len(X_train), mini_batchsize)):
            yd = (floatX(y_train[start:end])).reshape(mini_batchsize, 1)
            cost_v = train(X_train[start:end], yd)

        #Done this cost prediction needs to change
        #fin_cost_test = fcost(predict(X_test), floatX(y_test).reshape(len(y_test), 1))
        #fin_cost_train = fcost(predict(X_train), floatX(y_train).reshape(len(y_train), 1))
        fin_cost_test =MSE(predict(X_train),y_train)
        fin_cost_train = MSE(predict(X_test),y_test)
        test_costs.append(fin_cost_test)
        train_costs.append(fin_cost_train)
        #print i, fin_cost_test, fin_cost_train

    # print 'final b_o values'
    # print b_o.get_value()

    #fin_cost_test = fcost(predict(X_test), floatX(y_test).reshape(len(y_test), 1))
    #fin_cost_train = fcost(predict(X_train), floatX(y_train).reshape(len(y_train), 1))
    fin_cost_test=MSE(predict(X_test),y_test)
    fin_cost_train=MSE(predict(X_train),y_train)
    print 'Hwidth: {}, BatchSize: {}, L2reg: {},Train: {}, Test: {}'.format(hidden_width, mini_batchsize, L2reg,
                                                                            fin_cost_train, fin_cost_test)


    #Calculate RMS error with simple mean prediction
    test_mean=np.mean(y_test)
    train_mean=np.mean(y_train)

    mean_p_test=np.ones(y_test.size)*test_mean
    mean_p_train=np.ones(y_train.size)*train_mean

    #test_cost=fcost(floatX(mean_p_test).reshape(len(y_test), 1), floatX(y_test).reshape(len(y_test), 1))
    #train_cost=fcost(floatX(mean_p_train).reshape(len(y_train), 1), floatX(y_train).reshape(len(y_train), 1))
    test_cost=MSE(mean_p_test,y_test)
    train_cost=MSE(mean_p_train,y_train)

    tArray=np.ones(epochs)*test_cost
    #print 'MSE for mean prediction, Train:{} ,Test:{}'.format(train_cost,test_cost)

    plt.plot(range(epochs),test_costs,label='Test')
    plt.plot(range(epochs),train_costs,label='Train')
    plt.plot(range(epochs),tArray,label='Reference')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('NumTrainPoints: {}, TrainCost:{}, TestCost: {}'.format(numTrainPoints,fin_cost_train, fin_cost_test))
    #plt.show()
    plt.savefig('logs/exp4_TP{}.png'.format(numTrainPoints))
    plt.close()
    return fin_cost_train,fin_cost_test

if __name__ == "__main__":



    fin_cost_train,fin_cost_test=run_test(L2reg=0.01, hidden_width=10,numTrainPoints=10,mini_batchsize=5)
    fin_cost_train,fin_cost_test=run_test(L2reg=0.01, hidden_width=10,numTrainPoints=20,mini_batchsize=5)
    fin_cost_train,fin_cost_test=run_test(L2reg=0.01, hidden_width=10,numTrainPoints=30,mini_batchsize=5)
    fin_cost_train,fin_cost_test=run_test(L2reg=0.01, hidden_width=10,numTrainPoints=50,mini_batchsize=5)
    fin_cost_train,fin_cost_test=run_test(L2reg=0.01, hidden_width=10,numTrainPoints=100,mini_batchsize=5)
    fin_cost_train,fin_cost_test=run_test(L2reg=0.01, hidden_width=10,numTrainPoints=500,mini_batchsize=5)
    fin_cost_train,fin_cost_test=run_test(L2reg=0.01, hidden_width=10,numTrainPoints=2000,mini_batchsize=5)

