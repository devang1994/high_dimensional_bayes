__author__ = 'da368'

from load_synthetic import load_synthetic as load
import numpy as np

X_train, X_test, y_train, y_test = load()

print np.std(y_test)

print np.var(y_test)


#print np.std(y_train)
print np.mean(y_test)

print np.shape(y_test)[0]
print y_test.size