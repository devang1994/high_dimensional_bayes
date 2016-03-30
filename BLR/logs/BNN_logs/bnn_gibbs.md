![abc](2016-03-30 23.11.50.jpg)


```

for i in range(len(weights)):
            a=(np.sum(np.square(weights[i])) + np.sum(np.square(biases[i])))

            b=weights[i].size + biases[i].size
            scales[i]= 2.0/( (2.0/scales[i]) + a)
            shapes[i]= shapes[i] + b/2.0

            gamma_samples[i]=np.random.gamma(shapes[i],scales[i])

            print 'a {} , b {}'.format(a,b)
            print 'scales {}'.format(scales)
            print 'shapes {}'.format(shapes)
            print 'gamma {}'.format(gamma_samples)
            
            