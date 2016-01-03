##Uncertainty due to initialisation

Exp7 

```python
for rand_seed in range(20):
    np.random.seed(rand_seed)
    fin_cost_train, fin_cost_test=mlp_synthetic(L2reg=0.0001, numTrainPoints=2000,mini_batchsize=5)
    print 'Seed:{}, train:{}, test:{}'.format(rand_seed,fin_cost_train,fin_cost_test)
    
```

Explore the mean and Standard deviation 

For Vanilla MLP  (L2reg=0.0001, numTrainPoints=2000,mini_batchsize=5)

train mean 0.00380359066858  
train std 0.00386916729348  
train mean 0.0193062937606  
train std 0.0254937149911  

For Rand Proj (L2reg=0.0001, numTrainPoints=2000,mini_batchsize=5,proj_width=50)

train mean 0.278879095202  
train std 0.0202687507241  
train mean 0.644389849733  
train std 0.0607691570807  




exp7.log and exp7_randProj.log has raw data 

    