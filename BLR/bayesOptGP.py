__author__ = 'da368'

import numpy as np
import GPy
import matplotlib.pyplot as plt

np.random.seed(42)
def objective(x):
    return (np.sin(x * 7) + np.cos(x * 17))

# noise =0.2
# ntrain=150
# ntest =1000
#
# Xtrain = np.random.uniform(low=-1.0, high=2.0, size=(ntrain,1))
# ytrain = objective(Xtrain) + np.random.randn(ntrain,1) * noise
# xtest = np.linspace(-1., 2., ntest)
# xtest=  xtest.reshape(ntest,1)
#
# kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
# m = GPy.models.GPRegression(Xtrain,ytrain,kernel)
#
# print m
#
# m.optimize_restarts(num_restarts = 10)
# print m
# plt.figure(1)
# m.plot()
# #
# # plt.show()
#
# Yp, Vp = m.predict(xtest)
#
# Sp=np.sqrt(Vp)
# plt.figure(2)
# plt.plot(xtest,Yp)
# plt.plot(xtest,Yp-2*Sp)
# plt.plot(xtest,Yp+2*Sp)
#
#
# plt.show()
def acquisition_UCB(m, s, k=0.2):
    a = m - k * s
    return a


def bayes_opt(func,initial_random=2,k=0.2):
    '''function to do bayesOpt on and number of initial random evals
    noise is artificially added to objective function calls when training
    '''
    noise=0.2
    ntest=1000
    ntrain=initial_random #number of initial random function evals
    xtrain=np.random.uniform(low=-1.0, high=1.0, size=(ntrain,1))
    print xtrain.shape
    ytrain = objective(xtrain) + np.random.randn(ntrain,1) * noise
    print ytrain.shape
    xtest = np.linspace(-1., 2., ntest)
    xtest=  xtest.reshape(ntest,1)
    ytest = objective(xtest)
    plt.figure(1)                # the first figure

    # plt.plot(xtest, objective(xtest), color='black')
    # plt.plot(xtrain, ytrain, 'ro')

    for i in range(25):
        print 'it:{}'.format(i)

        kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m = GPy.models.GPRegression(xtrain,ytrain,kernel)

        m.optimize_restarts(num_restarts = 10)


        mu, s2 = m.predict(xtest)
        # mu, s2 = BLR(0.2, 4, train_transformed, test_transformed, ytrain)

        alpha=acquisition_UCB(mu, np.sqrt(s2),k=k)

        index=np.argmin(alpha)
        #TODO  use dradient descent instead of argmin, think about multi D

        next_query=xtest[index]
        next_y=objective(next_query) + np.random.randn(1,1) * noise



        plt.figure(i+2)                # the first figure
        s = np.sqrt(s2)  # 2 standard deviations




        plt.plot(xtest, objective(xtest), color='black',label='objective', linewidth=2.0)
        plt.plot(xtrain, ytrain, 'ro')
        plt.plot(xtest, mu, color='r', label='posterior')
        # plt.plot(xtest, mu - s, color='blue', label='credible')
        # plt.plot(xtest, mu + s, color='blue', label='interval')
        plt.plot(xtest,alpha,label='acquistion func',color='green')
        # plt.plot(xtest,np.zeros(ntest),color='black')
        plt.plot(xtest,s+2,label='sigma+2',color='blue')
        plt.title('BLR with learned features ,ntrain:{}'.format(xtrain.shape[0]))
        plt.legend(fontsize = 'x-small')
        plt.savefig('bayesOptGPNtrain{}k{}init{}.png'.format(xtrain.shape[0],k,ntrain),dpi=300)
        xtrain=np.vstack((xtrain,next_query))
        ytrain=np.vstack((ytrain,next_y))


if __name__ == '__main__':
    bayes_opt(objective,k=1.5,initial_random=10)
