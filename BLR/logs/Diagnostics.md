###Diagnostic tests on BLR + NN

Attempting to recreate Snoeks paper on BLR+NN

Objective func 

f(x) = sin(7x) +cos(17x)

Trained with 50 data points 

Figure below simply observes the performance of the NN on the data

![fig](figure_1.png)

Neural Net used 

1 -> 50 -> 50 -> 50 -> 1 

tanh non linearity 


---


###Performance of BLR

The bayesian linear regressor takes the activations of the NN as inputs to probabilistically model the function

![fig](figure_2.png)
