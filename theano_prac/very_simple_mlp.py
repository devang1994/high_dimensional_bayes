__author__ = 'Devang Agrawal'
"""Very simple MLP on MNIST
uses tutorial code"""

import theano
from theano import tensor as T
import numpy as np
from load import mnist


# from foxhound.utils.vis import grayscale_grid_vis, unit_scale
# from scipy.misc import imsave

def floatX(X):
    """convert to np array with floatX"""
    return np.asarray(X, dtype=theano.config.floatX)


def init_weights(shape):
    """theano shared variable with given shape"""
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def init_bias(n):
    """theano shared variable with given size"""
    return theano.shared(value=np.zeros((n,),dtype=theano.config.floatX))



def sgd(cost, params, lr=0.05):
    """same as simple update, just does it over all params"""
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates


def model(X, w_h, w_o,b_h,b_o):
    h = T.nnet.relu(T.dot(X, w_h)+b_h)
    pyx = T.nnet.softmax(T.dot(h, w_o)+b_o)
    return pyx


trX, teX, trY, teY = mnist(onehot=True)

X = T.fmatrix()
Y = T.fmatrix()

w_h = init_weights((784, 625))
w_o = init_weights((625, 10))
b_h= init_bias(625)
b_o=init_bias(10)

print T.pprint(b_o)
print b_o.get_value()
py_x = model(X, w_h, w_o,b_h,b_o)
y_x = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
params = [w_h, w_o,b_h,b_o]
updates = sgd(cost, params)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

for i in range(100):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    print "epoch: {}, accuracy: {}".format(i, np.mean(np.argmax(teY, axis=1) == predict(teX)))


print 'final b_o values'
print b_o.get_value()
