__author__ = 'da368'
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import numpy as np
def load_airfoil():
    '''returns the airfoil dataset after preprocessing
        all means are set to zero, all variances 1'''

    path='airfoil_self_noise.dat'
    dataset = np.loadtxt(open(path), delimiter="\t")

    X = dataset[:,0:5]
    y = dataset[:,5]

    X_scaled=preprocessing.scale(X)
    y_scaled=preprocessing.scale(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.33, random_state=42)

    return X_train,X_test,y_train,y_test
