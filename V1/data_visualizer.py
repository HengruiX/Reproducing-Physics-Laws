import numpy
import matplotlib.pyplot as plt
from data_generator import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

X1 = np.linspace(-2, 2, 500, endpoint=True)
X2 = np.linspace(-2, 2, 500, endpoint=True)

X1, X2 = np.meshgrid(X1, X2)
Y = func(X1, X2)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X1, X2, Y, cmap=cm.winter)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()