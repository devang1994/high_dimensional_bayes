import numpy as np
import GPy
import matplotlib.pyplot as plt

def objective(x):
    return (np.sin(x * 7) + np.cos(x * 17))

noise =0.2
ntrain=150

Xtrain = np.random.uniform(low=-1.0, high=2.0, size=(ntrain,1))
ytrain = objective(Xtrain) + np.random.randn(ntrain,1) * noise

# print Xtrain.shape
# print ytrain.shape
#
#
# Xtrain=np.random.uniform(-3.,3.,(20,1))
# ytrain=np.sin(Xtrain) + np.random.randn(20,1)*0.05
#
#
# print Xtrain.shape
# print ytrain.shape




kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
m = GPy.models.GPRegression(Xtrain,ytrain,kernel)

print m

m.optimize_restarts(num_restarts = 10)
print m

m.plot()

plt.show()