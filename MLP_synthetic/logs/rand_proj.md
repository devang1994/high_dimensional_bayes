#Random Projections

##Exp6

```python
def proj_matrix(shape):
scale = sqrt(6. / (shape[1] + shape[0]))
return floatX(np.random.uniform(low=-scale, high=scale, size=shape))
```
each element of random projection matrix chosen randomly from uniform distribution

Simple check

Without random projection

    mlp_synthetic(L2reg=0.0001, numTrainPoints=2000,proj_width=100,mini_batchsize=5)
    >>>NumTP: 2000, Hwidth: 10, BatchSize: 5, L2reg: 0.0001,Train: 0.00179861398386, Test: 0.0069205807329

    
    

With Random Projection, of same width 100->100

    mlp_synthetic_proj(L2reg=0.0001, numTrainPoints=2000,proj_width=100,mini_batchsize=5)
    >>>NumTP: 2000, Hwidth: 10, BatchSize: 5, L2reg: 0.0001,Train: 0.0135878321865, Test: 0.0552236782083




    
    
