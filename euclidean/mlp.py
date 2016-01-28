import theano
from theano import tensor as T
import numpy as np
from load_data import load_data as load
from math import sqrt
from adam import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as MSE
import cPickle as pickle
rand_seed=20
np.random.seed(rand_seed)

epochs = 1000
refError = 0.0638835188641


# this is the main code base

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


def mlp_synthetic(L2reg=0.01, hidden_width=10, mini_batchsize=5, numTrainPoints=2000):
    X_train, X_test, y_train, y_test = load()

    X_train = X_train[:numTrainPoints]
    y_train = y_train[:numTrainPoints]

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
    print 'NumTP: {}, Hwidth: {}, BatchSize: {}, L2reg: {}, Seed {},Train: {}, Test: {}'.format(numTrainPoints, hidden_width,
                                                                                       mini_batchsize, L2reg,rand_seed,
                                                                                       fin_cost_train, fin_cost_test)


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

    # plt.plot(range(epochs), test_costs, label='DatPts:{}'.format(numTrainPoints))
    # plt.plot(range(epochs),train_costs,label='Train')

    # plt.xlabel('Epochs')
    # plt.ylabel('Error')
    # plt.title('NumTrainPoints: {}, TrainCost:{}, TestCost: {}'.format(numTrainPoints,fin_cost_train, fin_cost_test))
    # plt.show()
    # plt.close()
    return fin_cost_train, fin_cost_test


def exp5_innerloop(L2reg=0.01, hidden_width=10, mini_batchsize=5):
    test_costs = []
    eval_pts = [100, 200,250, 300,350,400, 450,500, 700, 1000, 1500, 2000]
    for numTP in eval_pts:
        fin_cost_train, fin_cost_test = mlp_synthetic(L2reg=L2reg, hidden_width=hidden_width, numTrainPoints=numTP,
                                                      mini_batchsize=mini_batchsize)
        test_costs.append(fin_cost_test)

    plt.plot(eval_pts, test_costs, label='L2reg:{}'.format(L2reg))


def exp9(hidden_width=10, mini_batchsize=5):

    #like exp5 but with averaging
    eval_pts = [100, 200, 300,400,500,1000, 1500, 2000]
    for i in np.arange(-3,0):
        L2reg = pow(10, i)
        test_costs = []
        for numTP in eval_pts:
            temp=[]
            for k in range(5):
                fin_cost_train, fin_cost_test = mlp_synthetic(L2reg=L2reg, hidden_width=hidden_width, numTrainPoints=numTP,
                                                      mini_batchsize=mini_batchsize)
                temp.append(fin_cost_test)
            test_costs.append(np.mean(temp))#saving the mean of 5 tests as the new test cost

        plt.plot(eval_pts, test_costs, label='L2reg:{}'.format(L2reg))
    tArray = np.ones(2000) * refError
    plt.plot(range(2000), tArray, label='Reference', color='black', linewidth=2.0)
    plt.legend()
    plt.xlabel('Num Training Points')
    plt.ylabel('Error')
    plt.savefig('logs/exp9a.png', dpi=400)
    plt.show()

def exp10(L2reg=0.01,mini_batchsize=5):
    #for fixed L2 reg explore width of hidden network
    #savefig with amount of L2

    eval_pts = [100, 200, 300,400,500,1000, 1500, 2000]
    for hidden_width in [10]:
        test_costs = []
        for numTP in eval_pts:

            fin_cost_train, fin_cost_test = mlp_synthetic(L2reg=L2reg, hidden_width=hidden_width, numTrainPoints=numTP,
                                                  mini_batchsize=mini_batchsize)

            test_costs.append(fin_cost_test)#saving the mean of 5 tests as the new test cost

        plt.plot(eval_pts, test_costs, label='Hwidth:{}'.format(hidden_width))
    tArray = np.ones(2000) * refError
    plt.plot(range(2000), tArray, label='Reference', color='black', linewidth=2.0)
    plt.legend()
    plt.xlabel('Num Training Points')
    plt.ylabel('Error')
    plt.savefig('logs/exp12aL2reg{}.png'.format(L2reg), dpi=400)
    plt.show()

def exp5():
    #this is exp5 code
    for i in np.arange(-3, 1):
        L2reg = pow(10, i)
        exp5_innerloop(L2reg=L2reg)

    tArray = np.ones(2000) * refError
    plt.plot(range(2000), tArray, label='Reference', color='black', linewidth=2.0)
    plt.legend()
    plt.xlabel('Num Training Points')
    plt.ylabel('Error')
    plt.savefig('logs/exp8a.png', dpi=400)
    #plt.show()


if __name__ == "__main__":

    # for i in np.arange(-3, 3):
    #     L2reg = pow(10, i)
    #     exp5(L2reg=L2reg)
    #
    # tArray = np.ones(2000) * refError
    # plt.plot(range(2000), tArray, label='Reference', color='black', linewidth=2.0)
    # plt.legend()
    # plt.xlabel('Num Training Points')
    # plt.ylabel('Error')
    # plt.savefig('logs/exp5b.png', dpi=400)
    # plt.show()
    # train_costs=[]
    # test_costs=[]
    # for rand_seed in range(20):
    #     np.random.seed(rand_seed)
    #     fin_cost_train, fin_cost_test=mlp_synthetic(L2reg=0.0001, numTrainPoints=2000,mini_batchsize=5)
    #     train_costs.append(fin_cost_train)
    #     test_costs.append(fin_cost_test)
    #     #print 'Seed:{}, train:{}, test:{}'.format(rand_seed,fin_cost_train,fin_cost_test)
    #
    # print 'train mean', np.mean(train_costs)
    # print 'train std', np.std(train_costs)
    # print 'train mean', np.mean(test_costs)
    # print 'train std', np.std(test_costs)
    # with open("exp7.pickle", "wb") as f:
    #     pickle.dump((train_costs,test_costs), f)
    exp10(L2reg=0.0001)
    # exp10(L2reg=0.001)
    # exp10(L2reg=0.1)
    # mlp_synthetic(0.01,hidden_width=10)
