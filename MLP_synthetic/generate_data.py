import theano
from theano import tensor as T
import numpy as np
import cPickle as pickle
import gzip


def floatX(X):
    """convert to np array with floatX"""
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    """theano shared variable with given shape"""
    return theano.shared(floatX(np.random.randn(*shape) * 0.5))

def uniform_weights(shape, scale=0.5):
    return theano.shared(floatX(np.random.uniform(low=-scale, high=scale, size=shape)))

def init_bias(n):
    """theano shared variable with given size"""
    return theano.shared(value=np.zeros((n,), dtype=theano.config.floatX))

def model(X, w_h, w_o, b_h, b_o):
    h = T.nnet.relu(T.dot(X, w_h) + b_h)
    op = T.dot(h, w_o) + b_o
    return op

X = T.fmatrix(name='X')
input_size = 100
hidden_width = 10
#one hidden layer 100->10->1
w_h = uniform_weights((input_size, hidden_width)) #uniform from -0.5 to 0.5
w_o = uniform_weights((hidden_width, 1)) #uniform from -0.5 to 0.5

b_h = init_bias(hidden_width)#zero
b_o = init_bias(1)#zero

op = model(X, w_h, w_o, b_h, b_o)

create_datapoint = theano.function(inputs=[X], outputs=op, name='create_pt', allow_input_downcast=True)
nPoints = 2500
xp = np.random.uniform(low=-1.0, high=1.0, size=(nPoints, input_size))
#xp = np.random.rand(nPoints, input_size)
print xp.shape
print xp[1]
yp = create_datapoint(xp)

print yp[1]

print xp.shape
print yp.shape

pickle.dump(xp, gzip.open('synthetic_X.pkl.gz', 'wb'))
pickle.dump(yp, gzip.open('synthetic_y.pkl.gz', 'wb'))
