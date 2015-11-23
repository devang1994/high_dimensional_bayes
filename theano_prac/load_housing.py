__author__ = 'da368'
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import numpy as np
def load_housing():
    '''returns the airfoil dataset after preprocessing
        all means are set to zero, all variances 1'''

    path='housing.data'
    dataset = np.loadtxt(open(path))

    X = dataset[:,0:13]
    y = dataset[:,13]

    X_scaled=preprocessing.scale(X)
    print np.std(y)
    y_scaled=preprocessing.scale(y)
    print np.std(y_scaled)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42)

    return X_train,X_test,y_train,y_test
