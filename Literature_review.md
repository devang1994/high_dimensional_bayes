##Literature review

Snoek 2015 Scalable BO

Replace GP with DNN to model distribution over functions.

Use NN to learn adaptive set of basis functions for BLR

References for Bayesian NN replacing only the last layer.
 La ́zaro-Gredilla & Figueiras-Vidal (2010); Hinton & Salakhutdinov (2008) and Calandra et al. (2014a)
 
 
 Desired Characteristics of GP and DeepNet, flexibility and well calibrated uncertainty.
 
 **Adaptive basis regression**

Only treat output weights probabilistically and use point estimates otherwise
 
Z is the last hidden layer of NN

Train NN on inputs X->Z->Y

Use Z as our set of basis functions.
 
 
 ---
 
 No mention of Mapping from Z back to X