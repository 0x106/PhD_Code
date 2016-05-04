import matplotlib.pyplot as plt
import numpy as np
import random

def rbf(x, y, S):

	z = x-y
	q = np.dot(z.T, np.dot(S, z))

	return np.exp(-( q ))

def HSIC(X, Y, H, S, N):
	K, L  = np.zeros((N,N)), np.zeros((N,N))
	for i in range(N):
		for k in range(N):
			K[i,k] = rbf(X[:,i], X[:,k], S)
			L[i,k] = rbf(Y[:,i], Y[:,k], S)

	KH = np.dot(K,H)
	LH = np.dot(L,H)

	sf = np.sqrt((1./(N*N)) * np.trace(np.dot(LH, LH)))

	return np.sqrt((1./(N*N)) * np.trace(np.dot(KH, LH))) / sf

def test():
	N, D = 100, 2
	X = np.linspace(0, 4.*np.pi, N)

	mag1 = 1.2
	mag2 = 2.8
	freq1 = 1.0
	freq2 = 0.8

	target = np.zeros((D,N))
	Y = np.zeros((D,N))
	target[0,:] = mag1 * np.sin(freq1 * X)
	target[1,:] = mag2 * np.sin(freq2 * X)

	S = np.eye(D)
	S *= (1./(20.*D))

	H = np.zeros((N, N))
	for i in range(N):
		for k in range(N):
			H[i,k] = 0. - (1./N)
		H[i,i] = 1. - (1./N)

	t = 1.4
	t_mag1  = 1.2
	t_mag2  = 2.8 - t
	t_freq1 = 1.0
	t_freq2 = 0.8

	results = np.zeros(2*10)

	for i in range(2*10):
		
		Y[0,:] = t_mag1 * np.sin(t_freq1 * X)
		Y[1,:] = t_mag2 * np.sin(t_freq2 * X) 

		results[i] = HSIC(target, Y, H, S, N)

		# plt.plot(target[0,:])
		# plt.plot(target[1,:])
		# plt.plot(Y[0,:])
		# plt.plot(Y[1,:])

		print i, results[i], '', mag2, t_mag2 

		# plt.show()

		t_mag2 += 0.1

	plt.plot(results)
	plt.show()

test()


