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
import bnn_map as bnn_map
import math
import matplotlib.mlab as mlab


def objective(x):
    return (np.sin(x * 7/4) + np.cos(x * 17/4))

def make_nn_funs(layer_sizes, weight_prior_std=1.0, noise_var=0.1, nonlinearity=np.tanh):
    """These functions implement a standard multi-layer perceptron."""
    shapes = zip(layer_sizes[:-1], layer_sizes[1:])
    num_weights = sum((m+1)*n for m, n in shapes)

    def unpack_layers(weights):
        """weights is shape (num_weight_samples x num_weights)"""
        num_weight_sets = len(weights)
        for m, n in shapes:
            yield weights[:, :m*n]     .reshape((num_weight_sets, m, n)),\
                  weights[:, m*n:m*n+n].reshape((num_weight_sets, 1, n))
            weights = weights[:, (m+1)*n:]

    def predictions(weights, inputs):
        """weights is shape (num_weight_samples x num_weights)
           inputs  is shape (num_datapoints x D)"""
        inputs = np.expand_dims(inputs, 0)
        for W, b in unpack_layers(weights):
            outputs = np.einsum('mnd,mdo->mno', inputs, W) + b
            inputs = nonlinearity(outputs)
        return outputs

    def loglik(weights, noise_var, inputs, targets):
        preds = predictions(weights, inputs)
        log_lik_1 = - 1/2 * np.sum((preds - targets)**2, axis=1)[:, 0] / noise_var
        log_lik_2 = - 1/2 * np.log(2*math.pi*noise_var) * targets.shape[0] * targets.shape[1]
        return log_lik_1 + log_lik_2

    def gaussian_kl(w_mean, w_var, prior_var):
        kl = np.sum(np.log(prior_var) / 2.0 \
            - np.log(w_var) / 2.0 + (w_var + w_mean**2) / 2.0 / prior_var - 1 / 2.0)
        return kl

    def unpack_var_params(params):
        # variational dist is a diagonal Gaussian
        w_mean, log_w_std = params[:num_weights], params[num_weights:]
        w_std = np.exp(log_w_std)
        return w_mean, w_std

    rs = npr.RandomState(0)
    def variational_objective(params, prior_var, inputs, targets, N_train, num_samples):
        w_mean, w_std = unpack_var_params(params)
        samples = rs.randn(num_samples, num_weights) * w_std + w_mean
        kl_term = gaussian_kl(w_mean, w_std**2, prior_var)
        loglik_term = N_train / inputs.shape[0] * np.mean(loglik(samples, noise_var, inputs, targets))
        lower_bound = -kl_term + loglik_term
        # lower_bound = -kl_term
        # lower_bound = loglik_term
        # print(-kl_term)
        # print(loglik_term)
        return -lower_bound

    def get_error_and_ll(params, X, y, num_samples, location=0.0, scale=1.0):
        w_mean, w_std = unpack_var_params(params)
        noise_var_scale = noise_var * scale**2
        K = num_samples
        samples = rs.randn(K, num_weights) * w_std + w_mean
        outputs = predictions(samples, X) * scale + location
        log_factor = -0.5 * np.log(2 * math.pi * noise_var_scale) - 0.5 * (y - outputs)**2 / noise_var_scale
        ll = np.mean(logsumexp(log_factor - np.log(K), 0))
        pred_mean = np.mean(outputs, axis=0)
        error = np.sqrt(np.mean((y - pred_mean)**2))
        pred_var = np.var(outputs, axis=0)
        return pred_mean, pred_var, error, ll

    def prediction_test(params, X, num_samples, location=0.0, scale=1.0):
        w_mean, w_std = unpack_var_params(params)
        K = num_samples
        samples = rs.randn(K, num_weights) * w_std + w_mean
        outputs = predictions(samples, X) * scale + location
        pred_mean = np.mean(outputs, axis=0)
        pred_var = np.var(outputs, axis=0)
        return pred_mean, pred_var

    return num_weights, variational_objective, predictions, get_error_and_ll, unpack_layers, prediction_test, unpack_var_params


def fit_nn_reg(X, y, hidden_layer_sizes, batch_size, epochs, X_test, y_test, no_samples=20,
        mean_y_train=0.0, std_y_train=1.0, nonln='relu', weight_prior_std=1.0, noise_var=0.1, plot_toy=False, init_w=None):

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


    num_weights, elbo, predictions, get_error_and_ll, unpack_layers, prediction_test, unpack_params \
        = make_nn_funs(layer_sizes, nonlinearity=nonlinearity, weight_prior_std=weight_prior_std, noise_var=noise_var)
    elbo_grad = grad(elbo)
    prior_var = 1.0
    N_train = X.shape[ 0 ]
    

    print("    Epoch      |   train RMSE   |   test RMSE")

    if plot_toy:
        # Set up figure.
        fig = plt.figure(figsize=(12,8), facecolor='white')
        ax = fig.add_subplot(111, frameon=True)
        plt.show(block=False)
    def print_perf(epoch, w):
        num_samples_test = 500
        pred_mean, pred_var, rmse_train, ll = get_error_and_ll(w, X, y, location=0.0, scale=1.0, num_samples=num_samples_test)
        pred_mean, pred_var, rmse_test, ll = get_error_and_ll(w, X_test, y_test, location=0.0, scale=1.0, num_samples=num_samples_test)
        print("{0:15}|{1:15}|{2:15}|".format(epoch, rmse_train, rmse_test))

        if plot_toy:
            # # Plot data and functions.
            # plt.cla()
            # ax.plot(X.ravel(), y.ravel(), 'bx')
            # plot_inputs = np.reshape(np.linspace(-7, 7, num=300), (300,1))
            # outputs_mean, outputs_var = prediction_test(w, plot_inputs, num_samples_test)
            # ax.plot(plot_inputs, outputs_mean, 'b-')
            # ax.plot(plot_inputs, outputs_mean + 2*np.sqrt(outputs_var), 'b-')
            # ax.plot(plot_inputs, outputs_mean - 2*np.sqrt(outputs_var), 'b-')
            # ax.set_ylim([-1, 1])
            # plt.draw()
            # plt.pause(1.0/60.0)

            # Sample functions from posterior.
            rs = npr.RandomState(0)
            mean, std = unpack_params(w)
            #rs = npr.RandomState(0)
            sample_weights = rs.randn(10, num_weights) * std + mean
            plot_inputs = np.linspace(-7, 7, num=400)
            outputs = predictions(sample_weights, np.expand_dims(plot_inputs, 1))

            # Plot data and functions.
            plt.cla()
            ax.plot(X.ravel(), y.ravel(), 'bx')
            ax.plot(plot_inputs, outputs[:, :, 0].T)
            ax.set_ylim([-2, 3])
            plt.draw()
            plt.pause(1.0/60.0)


    # Train with adam
    batch_idxs = make_batches(X.shape[0], batch_size)

    # Initialize parameters
    rs = npr.RandomState(0)
    if init_w is None:
        init_mean = 0.1 * rs.randn(num_weights)
    else:
        init_mean = init_w

    init_log_std = -2 * np.ones(num_weights)
    init_var_params = np.concatenate([init_mean, init_log_std])
    w = init_var_params
    
    m1 = 0
    m2 = 0
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    alpha = 5e-3
    t = 0
    elbo_vec = []
    for epoch in range(epochs):
        permutation = np.random.choice(range(X.shape[ 0 ]), X.shape[ 0 ], replace = False)
        # print_perf(epoch, w)
        for idxs in batch_idxs:
            t += 1
            eb = elbo(w, weight_prior_std**2, X[ permutation[ idxs ] ], y[ permutation[ idxs ] ], N_train, num_samples=no_samples)
            elbo_vec.append(eb)
            print(eb)
            grad_w = elbo_grad(w, weight_prior_std**2, X[ permutation[ idxs ] ], y[ permutation[ idxs ] ], N_train, num_samples=no_samples)
            m1 = beta1 * m1 + (1 - beta1) * grad_w
            m2 = beta2 * m2 + (1 - beta2) * grad_w**2
            m1_hat = m1 / (1 - beta1**t)
            m2_hat = m2 / (1 - beta2**t)
            w -= alpha * m1_hat / (np.sqrt(m2_hat) + epsilon)
            t += 1

    print_perf(epochs-1, w)
    return w, get_error_and_ll, prediction_test, unpack_params, elbo_vec


def make_batches(N_data, batch_size):
    return [ slice(i, min(i + batch_size, N_data)) for i in range(0, N_data, batch_size) ]


def build_toy_dataset(func='cos', n_data=80, noise_std=0.1, D=1):
    rs = npr.RandomState(0)
    if func == 'cos':
        inputs  = np.concatenate([np.linspace(0, 4, num=n_data/2),
                                  np.linspace(5, 8, num=n_data/2)])
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
    elif func == 'sinPcos':
        inputs  = np.linspace(-4, 4, num=n_data)
        targets = objective(inputs) + rs.randn(n_data) * noise_std
        inputs  = inputs.reshape((len(inputs), D))
        # max_targets = np.max(np.abs(targets))
        targets = targets.reshape((len(targets), D)) / 2.0
        # normalizing

    return inputs, targets


if __name__ == '__main__':
    plot_toy = True

    N_data = 200
    noise_var = 0.01
    inputs, targets = build_toy_dataset(func='sinPcos', n_data=N_data, noise_std=np.sqrt(noise_var))
    # inputs, targets = build_toy_dataset(func='cube', n_data=200, noise_std=np.sqrt(noise_var))
    permutation = np.random.choice(range(inputs.shape[ 0 ]), inputs.shape[ 0 ], replace = False)
    X_train = inputs[permutation[:N_data/2], :]
    y_train = targets[permutation[:N_data/2], :]
    X_test = inputs[permutation[N_data/2:], :]
    y_test = targets[permutation[N_data/2:], :]
    n_hiddens = [50,50,50]
    mb_size = 20
    weight_prior_std=1.0
    noise_var=0.01

    no_epochs_map = 50
    w_map, log_prob_vec = bnn_map.fit_nn_reg(X_train, y_train, n_hiddens, mb_size, no_epochs_map, X_test, y_test,
        plot_toy=plot_toy, weight_prior_std=1.0, noise_var=noise_var)

    no_epochs_svi = 250
    w, get_error_and_ll, prediction_test, unpack_params, elbo_vec = fit_nn_reg(X_train, y_train, n_hiddens, mb_size, no_epochs_svi, 
        X_test, y_test, init_w=w_map, plot_toy=plot_toy, weight_prior_std=1.0, noise_var=noise_var)
    # w, get_error_and_ll, prediction_test, unpack_params, elbo_vec = fit_nn_reg(X_train, y_train, n_hiddens, mb_size, no_epochs_svi, 
    #     X_test, y_test, plot_toy=plot_toy, weight_prior_std=0.5, noise_var=noise_var)

    w_mean, w_std = unpack_params(w)
    plt.figure()
    plt.hist(w_mean, 50)
    plt.figure()
    plt.hist(20*np.log10(np.abs(w_mean/w_std)), 50)
    

    ncols = 5
    nrows = 4
    f, axarr = plt.subplots(nrows, ncols)
    for r in range(nrows):
        for c in range(ncols):
            ind = r*ncols + c
            xgauss = np.linspace(-2, 2, 200)
            ygauss = mlab.normpdf(xgauss, w_mean[ind], w_std[ind])
            axarr[r, c].plot(xgauss, ygauss, '-b')
            axarr[r, c].axvline(w_map[ind], color='r', linestyle='--')

    plt.figure()
    plt.plot(log_prob_vec)

    plt.figure()
    plt.plot(elbo_vec)

    plt.show()