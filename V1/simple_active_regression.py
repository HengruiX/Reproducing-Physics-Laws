import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from gplearn_mod.gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
import pickle

from data_generator import *

X, Y = get_synthetic_data_2D(func_nguyen)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

COMMITTEE_SIZE = 20
START_SIZE = 10
QUERY_SIZE = 5
ITER = 10
THRESH = 0.005

class QueryRegressor:
	def __init__(self, X_train, X_test, Y_train, Y_test, sr_param, verbose=0):
		self.X_train = X_train
		self.Y_train = Y_train
		self.X_test = X_test
		self.Y_test = Y_test
		self.sr_param = sr_param
		self.verbose = verbose

	def with_start(self, start):
		self.mask = np.full(len(X_train),True,dtype=bool)
		self.mask[start] = False
		self.X_labeled, self.Y_labeled = self.X_train[~self.mask], self.Y_train[~self.mask]
		return self

	def start_query(self):
		identified = False
		sample_sizes, scores = [], []

		for i in range(ITER):
			if not identified:
				sr = SymbolicRegressor(**self.sr_param)
				sr.fit(self.X_labeled, self.Y_labeled)

			score = sr.score(self.X_test, self.Y_test)

			if 1 - score < THRESH:
				identified = True

			if self.verbose > 0:
				print('[', i, ']', score)
			if self.verbose > 1:
				print('[', i, ']', sr._program)
				print()

			sample_sizes.append(len(self.X_labeled))
			scores.append(score)

			query = self.get_query(sr)
			print(query)
			self.mask[query] = False
			self.X_labeled, self.Y_labeled = self.X_train[~self.mask], self.Y_train[~self.mask]

		return sample_sizes, scores

	def get_query(self, sr):
		pass


class RandomQueryRegressor(QueryRegressor):
	def get_query(self, sr):
		avail = np.array(list(range(len(self.X_train))))
		avail = avail[self.mask]
		query = np.random.choice(avail, QUERY_SIZE, replace=False)
		return query

class ActiveQueryRegressor(QueryRegressor):
	def get_query(self, sr):
		avail = np.array(list(range(len(self.X_train))))
		avail = avail[self.mask]

		population = sr._programs[-1]
		fitness = [program.raw_fitness_ for program in population]
		committee = sorted(range(len(fitness)), key=lambda i: fitness[i])[:COMMITTEE_SIZE]

		predictions = np.array([population[c].execute(self.X_train[avail]) for c in committee]).T
		disagree = np.std(predictions, axis=1)
		query = avail[np.flip(np.argsort(disagree))[:QUERY_SIZE]]
		return query

		redraw = sorted(range(len(disagree)), key=lambda i: disagree[i], reverse=True)[:QUERY_SIZE]

random = []
active = []

fname = 'result/trial4'

for i in range(25):

	sr_param = {
		'population_size':1000, 
		'function_set':('add', 'sub', 'mul', 'sin', 'cos'), 
		'n_jobs':-1, 
		'generations':20, 
		'parsimony_coefficient':0.003
	}

	randomQueryRegressor = RandomQueryRegressor(X_train, X_test, Y_train, Y_test, sr_param, verbose=2)
	activeQueryRegressor = ActiveQueryRegressor(X_train, X_test, Y_train, Y_test, sr_param, verbose=2)

	start = np.random.choice(len(X_train), START_SIZE, replace=False)

	n, activei = activeQueryRegressor.with_start(start).start_query()
	n, randomi = randomQueryRegressor.with_start(start).start_query()
	

	random.append(randomi)
	active.append(activei)

	pickle.dump((n, random, active), open(fname, 'wb'))