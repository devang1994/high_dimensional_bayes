__author__ = 'da368'


from bayesNN_HMC_simple import unpack_theta

def test_unpack():
    hidden_width=20
    input_size=1
    output_size=1
    rng = numpy.random.RandomState(123)
    batchsize=1
    dim =hidden_width*(input_size+1)+(hidden_width+1)*output_size
    position = rng.randn(batchsize, dim).astype(theano.config.floatX)
    position = theano.shared(position)

    unpack_theta(position,hidden_width,input_size,output_size)