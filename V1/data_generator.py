import numpy as np

def func_nguyen(x1, x2):
	return np.sin(x1) + np.sin(x2**2)

def with_noise(x1, x2, func, std):
	return func(x1, x2) + np.random.normal(0, std)

def get_synthetic_data_2D(func, std=0):
	X1 = np.linspace(-3, 3, 500, endpoint=True)
	X2 = np.linspace(-3, 3, 500, endpoint=True)
	X = np.array([[x1, x2] for x1 in X1 for x2 in X2])
	Y = np.array([with_noise(x[0], x[1], func, std) for x in X]) 
	return X, Y

def get_airfoil_data():
	data = np.genfromtxt('data/airfoil_self_noise.dat', delimiter='\t')
	X = data[:, :-1]
	Y = data[:, -1]
	return X, Y