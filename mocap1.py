import matplotlib.pyplot as plt
import numpy as numpy
import random

def test():
	N = 100
	X = np.linspace(0, 4.*np.pi, N)

	Y = np.zeros((2,N))
	Y[0,:] = np.sin(X)
	Y[1,:] = np.sin(2.4*X)

	plt.plot(Y[0,:])
	plt.plot(Y[1,:])

	plt.show()



test()


