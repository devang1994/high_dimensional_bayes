__author__ = 'da368'
from sklearn import preprocessing
import numpy as np
import cPickle as pickle
import gzip
from sklearn.cross_validation import train_test_split


def load_synthetic():
    '''returns the airfoil dataset after preprocessing
        all means are set to zero, all variances 1'''

    X = pickle.load(gzip.open('synthetic_X.pkl.gz', 'rb'))
    y = pickle.load(gzip.open('synthetic_y.pkl.gz', 'rb'))

    #X = preprocessing.scale(X)
    #y = preprocessing.scale(y)
    print X.shape
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print X_train.shape
    print y_train.shape
    print X_test.shape
    print y_test.shape
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    load_synthetic()
