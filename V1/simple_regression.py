import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from gplearn_mod.gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
import pickle

from data_generator import *

X, Y = get_airfoil_data()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)


sr = SymbolicRegressor(
	population_size=1000, 
	verbose=1, 
	function_set=('add', 'sub', 'mul', 'sin', 'exp','sqrt', 'log', 'div', 'max'), 
	n_jobs=-1,
	generations=100, 
	parsimony_coefficient=0.0005
)

sr.fit(X_train, Y_train)
score = sr.score(X_test, Y_test)
print(score)
print(sr._program)

