__author__ = 'da368'
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import numpy as np
def load_blog():
    '''returns the airfoil dataset after preprocessing
        all means are set to zero, all variances 1'''

    path='/scratch-ssd/da368/datasets/blogData_train.csv'
    dataset = np.loadtxt(open(path,'rb'), delimiter=",",skiprows=1)

    # print dataset[0,:]
    # print 'fff'
    # print dataset[3,280]

    X = dataset[:,0:280]
    y = dataset[:,280]

    X=preprocessing.scale(X)
    y=preprocessing.scale(y)
    print X.shape
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X_scaled, y_scaled, test_size=0.33, random_state=42)

    X_train=X[0:10000,:]
    y_train=y[0:10000]
    X_test=X[10000:12000,:]
    y_test=y[10000:12000]
    # print X_train.shape
    # print y_train.shape
    # print X_test.shape
    return X_train,X_test,y_train,y_test

if __name__ =='__main__':
    load_blog()
