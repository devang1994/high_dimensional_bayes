##Euclidean Dataset

50 dimensional input X

1 dimensional output y = L2norm of X

````
input_size = 50
nPoints = 2000

def create_datapoint(xTrain):

    y=np.linalg.norm(xTrain,axis=1)
    y=np.reshape(y,(xTrain.shape[0],1))
    return y

xTrain = np.random.uniform(low=-1.0, high=1.0, size=(nPoints, input_size))
print xTrain.shape
yTrain = create_datapoint(xTrain)
````

####Reference Error

Just calculated on y_test

```
mean_pred=np.mean(y_test)
mean_arr=np.ones(y_test.shape)*mean_pred

print MSE(y_test,mean_arr)
```
same as code used for synthetic dataset

As expected good reference error was obtained


Ref_error=0.0638835188641



