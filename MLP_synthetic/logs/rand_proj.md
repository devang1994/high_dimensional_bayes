#Random Projections

##Exp6

```python
def proj_matrix(shape):
scale = sqrt(6. / (shape[1] + shape[0]))
return floatX(np.random.uniform(low=-scale, high=scale, size=shape))
```
each element of random projection matrix chosen randomly from uniform distribution

**Simple check**

Without random projection

    mlp_synthetic(L2reg=0.0001, numTrainPoints=2000,proj_width=100,mini_batchsize=5)
    >>>NumTP: 2000, Hwidth: 10, BatchSize: 5, L2reg: 0.0001,Train: 0.00179861398386, Test: 0.0069205807329

    
    

With Random Projection, of same width 100->100

    mlp_synthetic_proj(L2reg=0.0001, numTrainPoints=2000,proj_width=100,mini_batchsize=5)
    >>>NumTP: 2000, Hwidth: 10, BatchSize: 5, L2reg: 0.0001,Train: 0.0135878321865, Test: 0.0552236782083



The performance after random projection is expected to be the same but is in practise much worse


--

Exp5 was repeated with a random projection 100->50 

![Exp6](rand_proj1.png)


The performance is much worse than before, there is some hope with very low L2 reg

--

Repeat of last experiment with lower L2 regs and higher epochs

![Exp6](exp6b.png)

This showed much worse performance than before


--

Repeat with projection 100->80





--

Repeat with 100->30

--

Repeat of exp4 with evolution of graphs over epochs 


    
