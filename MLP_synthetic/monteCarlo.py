__author__ = 'da368'
import numpy as np
n=10000000
c=0
for i in range(n):
    x = np.random.uniform(0,1)
    y = np.random.uniform(0,1)
    if (((x-0.5)*(x-0.5)+(y-0.5)*(y-0.5))<0.25):
        c=c+1;



print (c*4.0)/(n)

print np.pi