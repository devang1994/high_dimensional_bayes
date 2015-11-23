import numpy

import theano

# This is the current suggested detect_nan implementation to
# show you how it work.  That way, you can modify it for your
# need.  If you want exactly this method, you can use
# ``theano.compile.monitormode.detect_nan`` that will always
# contain the current suggested version.

def detect_nan(i, node, fn):
    for output in fn.outputs:
        if (not isinstance(output[0], numpy.random.RandomState) and
            numpy.isnan(output[0]).any()):
            print '*** NaN detected ***'
            theano.printing.debugprint(node)
            print 'Inputs : %s' % [input[0] for input in fn.inputs]
            print 'Outputs: %s' % [output[0] for output in fn.outputs]
            break

x = theano.tensor.dscalar('x')
f = theano.function([x], [theano.tensor.log(x) * x],
                    mode=theano.compile.MonitorMode(
                        post_func=detect_nan))
f(0)  # log(0) * 0 = -inf * 0 = NaN