__author__ = 'da368'

import theano
import theano.tensor as T
import numpy as np
def model(X, w_h,b_h,w_o,b_o):
    #add relu
    h = T.nnet.relu(T.dot(X, w_h)+b_h)
    op = T.dot(h, w_o)+b_o
    return op

def init_bias(n):
    """theano shared variable with given size"""
    return theano.shared(value=np.ones((n,),dtype=theano.config.floatX))

def floatX(X):
    """convert to np array with floatX"""
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    """theano shared variable with given shape"""
    return theano.shared(floatX(np.ones(shape) * 1.0))


X = T.fmatrix(name='X')
Y = T.fmatrix(name='Y')#making this a matrix  might help

b_h=init_bias(2)

w_h=init_weights((2,2))

w_o=init_weights((2,1))

b_o=init_bias(1)

print(w_h.get_value())
print(b_h.get_value())
op=model(X,w_h,b_h,w_o,b_o)
print T.pprint(op)
#cost=T.mean()
cost=T.mean(T.sqr(op-Y))
xv=np.arange(10,dtype=np.float32).reshape(5,2)
yv=np.asarray([5,13,21,29,37],dtype=np.float32).reshape(5,1)
print xv
print(op.eval({X: xv}))
print(cost.eval({X:xv,Y:yv}))
print(yv)

