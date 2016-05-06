import  numpy as np

shape, scale = 1280, 0.00083534803555336968  # mean and dispersion
s = np.random.gamma(shape, scale, 10000)

print np.mean(s)
import matplotlib.pyplot as plt
import scipy.special as sps

count, bins, ignored = plt.hist(s, 100, normed=True)
y = bins**(shape-1)*(np.exp(-bins/scale) /
                     (sps.gamma(shape)*scale**shape))
plt.plot(bins, y, linewidth=2, color='r')
plt.show()