__author__ = 'da368'
import cPickle as pickle
import gzip


def load_synthetic():
    '''returns the airfoil dataset after preprocessing
        all means are set to zero, all variances 1'''

    X_train = pickle.load(gzip.open('synthetic_X_train.pkl.gz', 'rb'))
    y_train = pickle.load(gzip.open('synthetic_y_train.pkl.gz', 'rb'))
    X_test = pickle.load(gzip.open('synthetic_X_test.pkl.gz', 'rb'))
    y_test = pickle.load(gzip.open('synthetic_y_test.pkl.gz', 'rb'))


    # print X_train.shape
    # print y_train.shape
    # print X_test.shape
    # print y_test.shape
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    load_synthetic()
