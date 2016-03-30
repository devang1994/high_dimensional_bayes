def combinedGibbsHMC_BayesNN(n_samples, hWidths, X_train, y_train, scales,shapes):
    """

    :param n_samples:
    :param precisions:
    :param vy:
    :param hWidths:
    :param X_train:
    :param y_train:
    :param scales: params for gibbs . len is one more than precisions to sample vy as well
    :param shapes: params for gibbs
    :return:
    """

    input_size=X_train.shape[1]
    output_size=y_train.shape[1]

    # pick precisions etc from prior
    gamma_samples=[]
    for i in range(len(scales)):
        gamma_samples.append(np.random.gamma(shapes[i],scales[i]))
        print 'prior gamma_samples {}'.format(gamma_samples)



    train_err,test_err,samples,train_op_samples = sampler_on_BayesNN(burnin=10, n_samples=10, precisions=gamma_samples[0:(len(hWidths)+1)],
                                 vy=gamma_samples[len(hWidths)+1],X_train=X_train, y_train=y_train,hWidths=hWidths) # initial samples with big burnin

    num_sampled=10
    fin_samples=samples

    #  gibbs sampling
    last_train_op_sampled=train_op_samples[(train_op_samples.shape[0]-1), :].reshape(train_op_samples.shape[1],1)
    train_errs=[train_err]
    test_errs=[test_err]
#TODO study evolution of errors
    while num_sampled < n_samples:
        pass
        # first update params of gamma
        # then draw
        # TODO use this to debug, samples shape (10, 5251), train_op_samples (10, 100)
        # TODO Updates of gamma params is wrong debug


        #find weights and biases of last sample
        weights,biases=unpack_theta(samples,hWidths,input_size,output_size,index=(samples.shape[0]-1))

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

            # precisions on weights

        b = len(train_op_samples[(train_op_samples.shape[0]-1), :]) # length of train set
        a = np.sum(np.square(last_train_op_sampled-y_train))

        i=len(hWidths)+1
        scales[i]= 2.0/( (2.0/scales[i]) + a)
        shapes[i]= shapes[i] + b/2.0
        gamma_samples[i]=np.random.gamma(shapes[i],scales[i])

        train_err,test_err,samples,train_op_samples = sampler_on_BayesNN(burnin=10, n_samples=10, precisions=gamma_samples[0:(len(hWidths)+1)],
                                 vy=gamma_samples[len(hWidths)+1],X_train=X_train, y_train=y_train,hWidths=hWidths)


