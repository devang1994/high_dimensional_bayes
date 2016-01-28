import numpy as np
import cPickle as pickle
import gzip


input_size = 50

def create_datapoint(xTrain):

    y=np.linalg.norm(xTrain,axis=1)
    y=np.reshape(y,(xTrain.shape[0],1))
    return y

nPoints = 2000
xTrain = np.random.uniform(low=-1.0, high=1.0, size=(nPoints, input_size))
print xTrain.shape
yTrain = create_datapoint(xTrain)


print xTrain.shape
print yTrain.shape

nPoints = 500
xTest = np.random.uniform(low=-1.0, high=1.0, size=(nPoints, input_size))
print xTest.shape
yTest = create_datapoint(xTest)
print yTest.shape

pickle.dump(xTrain, gzip.open('euc_X_train.pkl.gz', 'wb'))
pickle.dump(yTrain, gzip.open('euc_y_train.pkl.gz', 'wb'))
pickle.dump(xTest, gzip.open('euc_X_test.pkl.gz', 'wb'))
pickle.dump(yTest, gzip.open('euc_y_test.pkl.gz', 'wb'))
