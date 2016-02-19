import numpy as np
import matplotlib.pyplot as plt

# x=np.linspace(-2,2,100)
# y=2*x
# f = plt.figure(1)
#
# plt.plot(x,y)
# f.show()
#
# g=plt.figure(2)
#
# y=3*(x)
#
# plt.plot(x,y)
#
# g.show()


# import matplotlib.pyplot as plt
plt.figure(1)                # the first figure
plt.subplot(211)             # the first subplot in the first figure
plt.plot([1, 2, 3])
plt.subplot(212)             # the second subplot in the first figure
plt.plot([4, 5, 6])


plt.figure(2)                # a second figure
plt.plot([4, 5, 6])          # creates a subplot(111) by default

plt.figure(1)                # figure 1 current; subplot(212) still current
plt.subplot(211)             # make subplot(211) in figure1 current
plt.title('Easy as 1, 2, 3') # subplot 211 title
plt.show()