__author__ = 'da368'
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import numpy as np
def load_crimes():
    '''returns the airfoil dataset after preprocessing
        all means are set to zero, all variances 1'''

    path='/scratch-ssd/da368/datasets/communities'
    dataset = np.loadtxt(open(path))
    #TODO split dataset asap rather than probabilistically
    #TODO figure how to deal with nin numbers in loadtxt

    X = dataset[:,4:13]
    y = dataset[:,20]

    X_scaled=preprocessing.scale(X)
    print np.std(y)
    y_scaled=preprocessing.scale(y)
    print np.std(y_scaled)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42)

    return X_train,X_test,y_train,y_test


if __name__ == '__main__':
    load_crimes()
