import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
def abc(s):

	plt.plot(np.ones(5)*s)
	plt.savefig('abc{}'.format(s))
	plt.close()

if __name__ == "__main__":

	abc(5)
	abc(6)


