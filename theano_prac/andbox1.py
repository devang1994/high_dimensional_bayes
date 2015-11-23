import theano
from theano import tensor as T
import numpy as np
from load_airfoil import load_airfoil
from detect_nan import detect_nan
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
    op = T.dot(h, w_o)+b_o
    return op


xv=np.arange(10,dtype=np.float32).reshape(5,2)
yv=np.asarray([5,13,21,29,37],dtype=np.float32).reshape(5,1)

X = T.fmatrix(name='X')
Y = T.fmatrix(name='Y')

w_h = init_weights((5, 2))
w_o = init_weights((2, 1))
b_h= init_bias(2)
b_o=init_bias(1)

op=model(X,w_h,w_o,b_h,b_o)
params = [w_h, w_o,b_h,b_o]

cost = (-1.0)*(T.mean(T.sqr(op - Y)))
updates = sgd(cost, params)

train = theano.function(inputs=[X, Y], outputs=cost,
                        updates=updates, allow_input_downcast=True,
                        name='train',mode=theano.compile.MonitorMode(
                        post_func=detect_nan))