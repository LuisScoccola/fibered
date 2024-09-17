import numpy as np
import pickle
from fibered.fibred import fibered, obstructionsData, geodesicDistance
from fibered.topologicalobstructions import sw1_obstruction
from ripser import ripser


def test_cylinder():
	# X stands for the attractor data, which consists of samples in Euclidean space
	# X_dm is the distance matrix for X
	# pi is a circular coordinate on the data, embedded in 2D
	with open("../data/cylinder-data.pckl", "rb") as handle:
		X, X_dm, pi = pickle.load(handle)
	k = 16
	d = 2
	e = 1
	fibred_embedding = fibered(X_dm, pi, k, e, d, fib_scale=1 / 10)
	dgms = ripser(fibred_embedding, n_perm=500)["dgms"]
	lifetimes = np.diff(dgms[1], axis=1)
	assert len(lifetimes[lifetimes > 1.4]) == 1, "dim(H1) of cylinder != 1"


def test_klein():
	# X stands for the Klein bottle portion of the cyclooctane data, which consists of samples in Euclidean space
	# X_dm is a distance matrix, computed as the shortest path distance in a 15-nn graph on X
	# pi is a circular coordinate on the data, embedded in 4D
	with open("../data/klein-data.pckl", "rb") as handle:
		X, X_dm, pi = pickle.load(handle)
	k = 16
	d = 3
	e = 1
	fibred_embedding = fibered(X_dm, pi, k, e, d, fib_scale=3 / 4)
	
	dm_embedding = geodesicDistance(fibred_embedding, k=15)
	dgms = ripser(dm_embedding, n_perm=300, maxdim=2, coeff=2, distance_matrix=True)["dgms"]
	lifetimes_H0 = np.diff(dgms[0], axis=1).ravel()
	lifetimes_H1 = np.diff(dgms[1], axis=1).ravel()
	lifetimes_H2 = np.diff(dgms[2], axis=1).ravel()
	assert np.sum(np.isinf(lifetimes_H0)) == 1, "Klein bottle in Z/2 should have one connected component"
	assert np.sum(lifetimes_H1 >= 0.75) == 2, "Klein in Z/2 should have two persistent H1 classes"
	assert np.sum(lifetimes_H2 >= 0.50) == 1, "Klein in Z/2 should have two persistent H1 classes"

	dgms = ripser(dm_embedding, n_perm=300, maxdim=2, coeff=3, distance_matrix=True)["dgms"]
	lifetimes_H0 = np.diff(dgms[0], axis=1).ravel()
	lifetimes_H1 = np.diff(dgms[1], axis=1).ravel()
	lifetimes_H2 = np.diff(dgms[2], axis=1).ravel()
	assert np.sum(np.isinf(lifetimes_H0)) == 1, "Klein bottle in Z/3 should have one connected component"
	assert np.sum(lifetimes_H1 >= 0.75) == 1, "Klein bottle in Z/3 should have 1 persistent H1 classes"
	assert np.sum(lifetimes_H2 >= 0.50) == 0, "Klein bottle in Z/3 should have no persistent H1 classes"

	cover_base, nerve_base, centers_base, omegas = obstructionsData(X_dm, pi, k, e, d)
	sol, basis, dth, dgms = sw1_obstruction(X_dm, nerve_base, centers_base, omegas)
	assert len(dgms[1]) == 1 and np.diff(dgms[1]).item() >= 0.25
