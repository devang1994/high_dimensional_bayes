from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
from autograd.scipy.misc import logsumexp
from autograd import grad
from autograd.core import grad, primitive
from autograd.scipy.special import gammaln


def make_nn_funs(layer_sizes, weight_prior_std=1.0, noise_var=0.1, nonlinearity=np.tanh):
    """These functions implement a standard multi-layer perceptron."""
    shapes = zip(layer_sizes[:-1], layer_sizes[1:])
    num_weights = sum((m+1)*n for m, n in shapes)

    def unpack_layers(weights):
        for m, n in shapes:
            cur_layer_weights = weights[:m*n]     .reshape((m, n))
            cur_layer_biases  = weights[m*n:m*n+n].reshape((1, n))
            yield cur_layer_weights, cur_layer_biases
            weights = weights[(m+1)*n:]

    def predictions(params, inputs):
        weight_list = unpack_layers(params)
        for W, b in weight_list:
            outputs = np.dot(inputs, W) + b
            inputs = nonlinearity(outputs)
        return outputs

    def logprob(params, inputs, targets, Ntrain):
        log_prior = np.sum(norm.logpdf(params, 0, weight_prior_std))
        preds = predictions(params, inputs)
        log_lik = np.sum(norm.logpdf(preds, targets, np.sqrt(noise_var)))
        return log_prior + Ntrain * 1.0 / inputs.shape[0] * log_lik

    def get_error(params, X, y, location=0.0, scale=1.0):
        outputs = predictions(params, X)
        outputs = outputs * scale + location
        outputs = outputs.reshape((y.shape[0], y.shape[1]))
        return np.sqrt( np.mean( (outputs - y)**2 ) ) 

    return num_weights, predictions, logprob, get_error


def fit_nn_reg(X, y, hidden_layer_sizes, batch_size, epochs, X_test, y_test, 
        mean_y_train=0.0, std_y_train=1.0, nonln='relu', weight_prior_std=1.0, noise_var=0.1, plot_toy=False):

    layer_sizes = np.array([ X.shape[ 1 ] ] + hidden_layer_sizes + [ 1 ])
    if nonln == 'tanh':
        nonlinearity = np.tanh
    elif nonln == 'relu':
        nonlinearity = lambda x: np.maximum(x, 0.0)
    elif nonln == 'rbf':
        nonlinearity = lambda x: norm.pdf(x, 0, 1)
    elif nonln == 'sin':
        nonlinearity = lambda x: np.sin(x)
    elif nonln == 'sigmoid':
        nonlinearity = lambda x: 1 / (1 + np.exp(-x))


    num_weights, predictions, logprob, get_error \
        = make_nn_funs(layer_sizes, nonlinearity=nonlinearity, weight_prior_std=weight_prior_std, noise_var=noise_var)
    logprob_grad = grad(logprob)
    Ntrain = X.shape[0]

    print("    Epoch      |   train RMSE   |   test RMSE")

    if plot_toy:
        # Set up figure.
        fig = plt.figure(figsize=(12,8), facecolor='white')
        ax = fig.add_subplot(111, frameon=True)
        plt.show(block=False)
        
    def print_perf(epoch, w):
        rmse_train = get_error(w, X, y, location=0.0, scale=1.0)
        rmse_test = get_error(w, X_test, y_test, location=0.0, scale=1.0)
        print("{0:15}|{1:15}|{2:15}|".format(epoch, rmse_train, rmse_test))

        if plot_toy:
            # Plot data and functions.
            plt.cla()
            ax.plot(X.ravel(), y.ravel(), 'bx')
            plot_inputs = np.reshape(np.linspace(-7, 7, num=300), (300,1))
            outputs = predictions(w, plot_inputs)
            ax.plot(plot_inputs, outputs)
            ax.set_ylim([-1, 1])
            plt.draw()
            plt.pause(1.0/60.0)


    # Train with adam
    batch_idxs = make_batches(X.shape[0], batch_size)

    # Initialize parameters
    rs = npr.RandomState(0)
    init_weights = 0.1 * rs.randn(num_weights)
    w = init_weights
    N_test = X_test.shape[0]


    m1 = 0
    m2 = 0
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    alpha = 1e-2
    t = 0
    log_prob_vec = []
    for epoch in range(epochs):
        permutation = np.random.choice(range(X.shape[ 0 ]), X.shape[ 0 ], replace = False)
        print_perf(epoch, w)
        for idxs in batch_idxs:
            t += 1
            lp = logprob(w, X[ permutation[ idxs ] ], y[ permutation[ idxs ] ], X.shape[ 0 ])
            log_prob_vec.append(lp)
            grad_w = logprob_grad(w, X[ permutation[ idxs ] ], y[ permutation[ idxs ] ], X.shape[ 0 ])
            m1 = beta1 * m1 + (1 - beta1) * grad_w
            m2 = beta2 * m2 + (1 - beta2) * grad_w**2
            m1_hat = m1 / (1 - beta1**t)
            m2_hat = m2 / (1 - beta2**t)
            w += alpha * m1_hat / (np.sqrt(m2_hat) + epsilon)
            t += 1

    return w, np.array(log_prob_vec)


def make_batches(N_data, batch_size):
    return [ slice(i, min(i + batch_size, N_data)) for i in range(0, N_data, batch_size) ]


def build_toy_dataset(func='cos', n_data=80, noise_std=0.1, D=1):
    rs = npr.RandomState(0)
    if func == 'cos':
        inputs  = np.concatenate([np.linspace(0, 3, num=n_data/2),
                                  np.linspace(6, 8, num=n_data/2)])
        targets = np.cos(inputs) + rs.randn(n_data) * noise_std
        inputs = (inputs - 4.0) / 2.0
        inputs  = inputs.reshape((len(inputs), D))
        targets = targets.reshape((len(targets), D)) / 2.0
    elif func == 'cube':
        inputs  = np.linspace(-3, 3, num=n_data)
        targets = inputs**3 + rs.randn(n_data) * noise_std
        inputs  = inputs.reshape((len(inputs), D))
        max_targets = np.max(np.abs(targets))
        targets = targets.reshape((len(targets), D)) / 2.0 / max_targets
    return inputs, targets


if __name__ == '__main__':
    plot_toy = True

    # inputs, targets = build_toy_dataset(func='cos', n_data=200, noise_std=np.sqrt(0.1))
    inputs, targets = build_toy_dataset(func='cube', n_data=200, noise_std=np.sqrt(0.1))
    permutation = np.random.choice(range(inputs.shape[ 0 ]), inputs.shape[ 0 ], replace = False)
    X_train = inputs[permutation[:100], :]
    y_train = targets[permutation[:100], :]
    X_test = inputs[permutation[100:], :]
    y_test = targets[permutation[100:], :]
    w, log_prob_vec = fit_nn_reg(X_train, y_train, [50], 20, 200, X_test, y_test, plot_toy=plot_toy)


    plt.figure()
    plt.hist(w)
    plt.figure()
    plt.plot(log_prob_vec)
    plt.show()