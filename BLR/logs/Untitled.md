##DNGO stuff

Using 3 hidden layers with tanh, similar to Snoek 2015

```
def model(X, w_h1,w_h2,w_h3, w_o, b_h1,b_h2,b_h3, b_o):
    h1 = T.nnet.tanh(T.dot(X, w_h1) + b_h1)
    h2 = T.nnet.tanh(T.dot(X, w_h2) + b_h2)
    h3 = T.nnet.tanh(T.dot(X, w_h3) + b_h3)
    op = T.dot(h3, w_o) + b_o
    return op
```

3 tanh non-linearitites 
    
    