__author__ = 'da368'
from load_synthetic import load_synthetic as load
import numpy as np
from sklearn.metrics import mean_squared_error as MSE

X_train, X_test, y_train, y_test = load()

#find reference error on the test dataset


mean_pred=np.mean(y_test)
mean_arr=np.ones(y_test.shape)*mean_pred

print MSE(y_test,mean_arr)

