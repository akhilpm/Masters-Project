'''

computes an unsupervised combination of multiple kernels
Author : Akhil P M

Kernel types
	1. Arc cosine
	2. Gaussian 
	3. Polynomial
	4. Sigmoid

'''

from settings import *


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


def arc_cosine_vector(X, Y, n_l):
	
	M = np.dot(X, Y.T)
	temp1 = np.diag(np.dot(X, X.T))
	temp2 = np.diag(np.dot(Y, Y.T))

	norm_matrix = np.outer(temp1,temp2) #the matix of k_xx, and K_yy's
	theta = np.arccos( np.maximum( np.minimum(M/np.sqrt(norm_matrix), 1.0), -1.0))
	M = np.multiply(np.power(norm_matrix, n_l/2.0), compute_J(n_l, theta)) / np.pi

	return M


def arc_cosine(X, Y, n_l):
	lenX = X.shape[0]
	incr = 1000
	M = np.zeros((lenX, Y.shape[0]))
	for i in range(0,lenX,incr):
		M[i:i+incr] = arc_cosine_vector(X[i:i+incr], Y, n_l)

	return M


def kernel(k_type, X, Y, param):
	""" computes the gram matrix for different kernels """

	if k_type==1:
		return arc_cosine(X, Y, param)
	elif k_type==2:
		return rbf_kernel(X, Y, gamma=param)
	else:
		return polynomial_kernel(X, Y, degree=param, gamma=0.1)


def processInput(i, K_T1, K_T2, flag):
	global D, M, P, matP, vecQ, gamma

	ele1 = np.outer(K_T1[i], K_T2[i])
	ele2 = np.outer(D[i], D[i])
	matP[j, k] += np.sum(np.multiply( np.multiply(ele1, ele2), P))

	if flag==1:
		temp = ( 2*gamma*np.multiply(M[i], D[i]) - 2*np.multiply(P[i], D[i]) )
		vecQ[k] += np.dot(temp, K_T2[i])

def getUMKL_coefficients(trainX, k_type, param):


	global D, M, P, matP, vecQ, gamma
	#set the timer
	start = time.time()

	n = trainX.shape[0]
	n_kernels = len(param)
	gamma = 0.01

	nbrs = NearestNeighbors(n_neighbors=30, algorithm='ball_tree', n_jobs=-1).fit(trainX)
	D = nbrs.kneighbors_graph(trainX).toarray()
	M = pow(euclidean_distances(trainX, trainX), 2)
	P = np.dot(trainX, np.transpose(trainX))


	matP = np.zeros((n_kernels, n_kernels))
	vecQ = np.zeros((n_kernels, 1))


	#this requires only n_kernels * n_kernels no of times kernel evaluation
	flag = 1
	for j in xrange(n_kernels):
		print('\nprocesing kernel %d started' %(j+1))
		print('starting time ' + str(datetime.now().hour) + ':' + str(datetime.now().minute))
		K_T1 = kernel(k_type[j], trainX, trainX, param[j])
		print('outer kernel %d evaluated at %d:%d' %(j+1, datetime.now().hour, datetime.now().minute))

		for k in xrange(n_kernels):
			K_T2 = kernel(k_type[k], trainX, trainX, param[k])
			print('inner kernel %d evaluated at %d:%d' %(k+1, datetime.now().hour, datetime.now().minute))

			Parallel(n_jobs=6)(delayed(processInput)(i, K_T1, K_T2, flag) for i in xrange(n))

		print('inner kernel '+str(k+1)+ ' completed at ' + str(datetime.now().hour) + ':' + str(datetime.now().minute))

		flag = 0
		print('\nprocesing kernel %d completed' %(j+1))
		print('completion time ' + str(datetime.now().hour) + ':' + str(datetime.now().minute))


	np.delete(P)

	print('QP solving started after %f Minutes\n' %((time.time()-start)/60))
	P = cvxopt.matrix(matP)
	q = cvxopt.matrix(vecQ)
	A = cvxopt.matrix(np.ones((1, n_kernels)))
	b = cvxopt.matrix(1.0)
	G = cvxopt.matrix(np.diag(np.ones(n_kernels) * -1))
	h = cvxopt.matrix(np.zeros(n_kernels))

	#solve the convex QP problem
	solvers.options['maxiters'] = 1000
	solution = solvers.qp(P, q, G, h, A, b)

	#obtain the solution mu
	mu = np.ravel(solution['x'])
	print('QP solving Time : %f Minutes\n' %((time.time()-start)/60))

	return mu



def getUMKL_gram_matrix(X, Y, k_type, param, mu):
	""" returrns the gram matrix after adding all kernels"""

	n_kernels = len(param)
	n1 = X.shape[0]
	n2 = Y.shape[0]
	K = np.zeros((n1, n2))

	for i in xrange(n_kernels):
		K += mu[i] * kernel(k_type[i], X, Y, param[i])

	return K
	

if __name__ == '__main__':
	main()

