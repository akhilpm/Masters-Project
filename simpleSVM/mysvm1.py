'''
SVM with standard QP solvers(cvxopt).
Author : Akhil P M
Notes:
	1. This doesn't scale well for large datasets as no of datapoints increases Gram Matrix will be huge.
'''


import numpy as np
import cvxopt
import time
from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from cvxopt import solvers
#from cvxopt import base, blas, lapack, solvers, matrix


def compute_J(N, theta):

  if N == 0:
    return np.pi - theta
  elif N == 1:
    return np.sin(theta) + (np.pi - theta) * np.cos(theta)
  elif N == 2:
    return 3*np.sin(theta)*np.cos(theta) + (np.pi - theta)*(1 + 2*pow(np.cos(theta), 2))
  elif N == 3:
    return 4*pow(np.sin(theta), 3) + 15*np.sin(theta)*pow(np.cos(theta), 2) + \
    (np.pi- theta)*(9*pow(np.sin(theta),2)*np.cos(theta) + 15*pow(np.cos(theta),3))
  else:    
    return np.zeros(theta.shape)


def arc_cosine(X, Y):
  """param = a vector of n(degree) values at each layer """
  param = np.array([0])
  no_of_layers = len(param)


  M = np.dot(X, Y.T)
  temp1 = np.diag(np.dot(X, X.T))
  temp2 = np.diag(np.dot(Y, Y.T)) 

  for i in xrange(no_of_layers):
    norm_matrix = np.outer(temp1,temp2) #the matix of k_xx, and K_yy's
    theta = np.arccos( np.maximum( np.minimum(M/np.sqrt(norm_matrix), 1.0), -1.0))
    n_l = param[i]
    M = np.multiply(np.power(norm_matrix, n_l/2.0), compute_J(n_l, theta)) / np.pi


    if i < no_of_layers-1:
      zero1 = np.zeros(len(temp1))
      zero2 = np.zeros(len(temp2))
      temp1 = np.multiply(np.power(temp1, n_l), compute_J(n_l, zero1)) / np.pi
      temp2 = np.multiply(np.power(temp2, n_l), compute_J(n_l, zero2)) / np.pi


  return M


def main():
	start = time.time()
	iris = datasets.load_iris()
	X = iris.data
	Y = iris.target

	#make it to a binary classification problem by labelling with -1 & +1
	Y[0:50] = 1.0
	Y[50:] = -1.0

	trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.3)
	trainX = trainX.astype(dtype = 'float32')
	testX = testX.astype(dtype = 'float32')
	n_samples = trainX.shape[0]


	min_max_scaler = preprocessing.MinMaxScaler()
	trainX = min_max_scaler.fit_transform(trainX)

	#compute the full dense gram matrix (Note: this doesn't scale well for large datasets)
	K = arc_cosine(trainX, trainX)

	P = cvxopt.matrix(np.outer(trainY,trainY) * K)
	q = cvxopt.matrix(np.ones(n_samples) * -1)
	A = cvxopt.matrix(trainY, (1,n_samples))
	A = A*1.0 #to make A float type
	b = cvxopt.matrix(0.0)
	G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
	h = cvxopt.matrix(np.zeros(n_samples))

	#solve the QP problem
	solvers.options['maxiters'] = 1000
	solution = solvers.qp(P, q, G, h, A, b)

	#solution alpha
	alpha = np.ravel(solution['x'])
	print alpha
	print solution['status']
	print('Execution Time :%f Seconds\n' %(time.time()-start))

if __name__ == '__main__':
	main()	
