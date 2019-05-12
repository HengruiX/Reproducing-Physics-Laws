import pickle
import matplotlib.pyplot as plt
import numpy as np

fname = 'result/'
n, A1, A2 = pickle.load(open(fname, 'rb'))

means1 = np.mean(A1, axis=0)
stds1 = np.std(A1, axis=0)/np.sqrt(len(A1))
means2 = np.mean(A2, axis=0)
stds2 = np.std(A2, axis=0)/np.sqrt(len(A2))

plt.errorbar(n, means1, yerr=stds1, label='random query')
plt.errorbar(n, means2, yerr=stds2, label='active query')
plt.xlabel('data queried')
plt.ylabel(r'$R^2$')
plt.legend()
plt.show()
