###Diagnostic Tests

18 December 2015

```python
X_train, X_test, y_train, y_test = load()

print np.std(y_test)

>>>0.854211436961

print np.std(y_test)

>>>0.729677179036

```

Note: In previous experiments the actual cost including L2 reg penalty factors has been quoted, this should be fixed to MSE, to avoid errors

--



####Mean Prediction 

Mean prediction was implemented 

```python
test_mean=np.mean(y_test)
train_mean=np.mean(y_train)

mean_p_test=np.ones(y_test.size)*test_mean
mean_p_train=np.ones(y_train.size)*train_mean

test_cost=MSE(mean_p_test,y_test)
train_cost=MSE(mean_p_train,y_train)
print  'Mean_pred,Train:{} ,Test:{}'.format(train_cost,test_cost)

>>>Mean_pred,Train:0.767789625154 ,Test:0.729677179036

```

These values are almost identical to 

Hwidth: 10, BatchSize: 100, L2reg: 1,Train: 0.767793228344, Test: 0.730858657727

which means that the current model is not doing much better than mean prediction
Using entire Dataset  
Parameters:

    L2reg=1, hidden_width=10, mini_batchsize=100
   

--

####Hyperparameter Search

Varying L2 and width of Hidden Layer , entire dataset

Best value obtained

    Hwidth: 50, BatchSize: 100, L2reg: 0.01,Train: 0.0310097190959, Test: 0.0734797714539

In general L2reg = 0.01 gives good results
details in exp3.log  

**Use L2reg=0.01 and Hwidth = 10 and redo previous experiments**

Note:L2reg>100 gives an unexpected error, possibly due to large values

--

