import  numpy as np

a = 0.05
shape, scale = 80 * a, 0.125 / a  # mean and dispersion
s = np.random.gamma(shape, scale, 10000)

print np.mean(s)
print np.std(s)
import matplotlib.pyplot as plt
import scipy.special as sps

count, bins, ignored = plt.hist(s, 100, normed=True)
y = bins**(shape-1)*(np.exp(-bins/scale) /
                     (sps.gamma(shape)*scale**shape))
plt.plot(bins, y, linewidth=2, color='r')
plt.show()