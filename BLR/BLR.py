__author__ = 'da368'
import numpy as np
import matplotlib.pyplot as plt


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








if __name__== '__main__':
    #each column is one datapoint
    #mapping from 10d to 1d
    #200 train data_points , 10 d each. 50 test pts
    #each row is basis functions belonging to one datapoint
    # phi = N x M
    # phi_train=np.ones(2000).reshape(200,10)
    # y_train=np.ones(200).reshape(200,1)
    # phi_test=np.arange(500).reshape(50,10)
    # y_test=np.ones(50).reshape(50,1)
    # a=np.dot(np.transpose(phi_train),phi_train)

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

