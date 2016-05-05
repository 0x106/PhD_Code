import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.optimize

def rbf(x, y, S):

	z = x-y
	q = np.dot(z.T, np.dot(S, z))

	return np.exp(-( q ))

def HSIC(params, N, D, H, KH, S, index):

	Y = np.zeros((D,N))

	Y[0,:] = params[0] * np.sin(params[1] * index)
	Y[1,:] = params[2] * np.sin(params[3] * index) 

	L  = np.zeros((N,N))
	for i in range(N):
		for k in range(N):
			L[i,k] = rbf(Y[:,i], Y[:,k], S)

	LH = np.dot(L,H)

	sf = np.sqrt((1./(N*N)) * np.trace(np.dot(LH, LH)))

	return -1 * ((1./(N*N)) * np.trace(np.dot(KH, LH)) / sf)

# def HSIC(X, Y, H, S, N):
# 	K, L  = np.zeros((N,N)), np.zeros((N,N))
# 	for i in range(N):
# 		for k in range(N):
# 			K[i,k] = rbf(X[:,i], X[:,k], S)
# 			L[i,k] = rbf(Y[:,i], Y[:,k], S)

# 	KH = np.dot(K,H)
# 	LH = np.dot(L,H)

# 	sf = np.sqrt((1./(N*N)) * np.trace(np.dot(LH, LH)))

# 	return (1./(N*N)) * np.trace(np.dot(KH, LH)) / sf

def test():
	N, D = 100, 2
	index = np.linspace(0, 4.*np.pi, N)

	x = np.zeros(4)
	x[0], x[1], x[2], x[3] = 1.2, 1.0, 2.8, 0.8

	target = np.zeros((D,N))
	Y = np.zeros((D,N))
	target[0,:] = x[0] * np.sin(x[1] * index)
	target[1,:] = x[2] * np.sin(x[3] * index)

	S = np.eye(D)
	S *= (1./(20.*D))

	H, K = np.zeros((N, N)), np.zeros((N,N))
	for i in range(N):
		for k in range(N):
			H[i,k] = 0. - (1./N)
			K[i,k] = rbf(target[:,i], target[:,k], S)
		H[i,i] = 1. - (1./N)

	KH = np.dot(K,H)

	init = np.zeros(4)
	init[0] = x[0] - 0.2
	init[1] = x[1] - 0.2
	init[2] = x[2] - 0.1
	init[3] = x[3] - 0.0
	result = scipy.optimize.minimize(HSIC, init, args=(N,D,H,KH,S,index), method='Nelder-Mead', options = {'disp':False})

	for i in range(x.shape[0]):
		print init[i], x[i], round(result.x[i], 3)

	# t = 1.4
	# t_mag1  = 1.2 - 0.1
	# t_mag2  = 2.8 - t
	# t_freq1 = 1.0
	# t_freq2 = 0.8

	# results = np.zeros(2*100)

	# for i in range(2*100):
		
	# 	Y[0,:] = t_mag1 * np.sin(t_freq1 * X)
	# 	Y[1,:] = t_mag2 * np.sin(t_freq2 * X) 

	# 	results[i] = HSIC(target, Y, H, S, N)

	# 	print i, results[i], '', mag2, t_mag2, mag1, t_mag1  

	# 	t_mag2 += 0.01

	# plt.plot(results)
	# plt.show()

test()


