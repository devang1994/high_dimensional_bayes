__author__ = 'da368'

'''Lets say inputs x are D-dimensional and outputs y are 1-dimensional.

Recall the idea is to learn a MLP from x to z, (let's just code a one hidden layer neural
network for now), where z has dimension 20 for example.
More explicitly, we can have the following:

h = ReLU( W1 * x + b1 )
z = W2 * h + b2

where * is a matrix product, W1, W2, b1, b2 are parameters to be learnt.

Now we map from z to y using a Gaussian process, with a Squared
Exponential kernel, with length scales and amplitude all fixed to 1.

You should be able to write a likelihood of y given x for this model, and this is
what we will optimize with respect to the parameters.
The nice thing is that once you write down the likelihood, Theano can compute gradients automatically.'''

import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

x = T.vector('x')
W1 = T.matrix('W1')
b1 = T.vector('b1')
W2 = T.matrix('W2')
b2 = T.vector('b2')

h = T.nnet.relu(T.dot(W1, x) + b1)
z = T.dot(W2, h) + b2

print(theano.pp(h))
