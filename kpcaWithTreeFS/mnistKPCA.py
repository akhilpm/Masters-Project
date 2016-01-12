'''
KPCA based feature engineering for MNIST handwritten digits classification
Author : Akhil P M
Kernel used : Arc-cosine Kernel
'''

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import svm, datasets
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets.mldata import fetch_mldata
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split



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


def arc_cosine_vector(X, Y):
	"""param = a vector of n(degree) values at each layer """
	param = np.array([1,1,1])
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

def arc_cosine(X, Y):
	lenX = X.shape[0]
	incr = 1000
	M = np.zeros((lenX, Y.shape[0]))
	for i in range(0,lenX,incr):
		M[i:i+incr] = arc_cosine_vector(X[i:i+incr], Y)

	return M	



def main():

	#set the timer
	start = time.time()

	#load the data
	mnist = fetch_mldata('MNIST original')
	mnist.target = mnist.target.astype(np.int32)

	seed = np.random.randint(1,30000)
	rand = np.random.RandomState(seed)
	items = len(mnist.target)
	indices = rand.randint(items, size = 70000)
	trindex = indices[0:30000]
	tsindex = indices[30000:]

	#scale down features to the range [0, 1]
	mnist.data = mnist.data/255.0
	mnist.data = mnist.data.astype(np.float32)

	trainX = mnist.data[trindex]
	testX = mnist.data[tsindex]
	trainY = mnist.target[trindex]
	testY = mnist.target[tsindex]

	#extract the features using KPCA
	kpca = KernelPCA(kernel='precomputed')
	kpca_train = arc_cosine(trainX[0:1000], trainX[0:1000])
	#Fit the model from data in X
	kpca.fit(kpca_train)

	kernel_train = arc_cosine(trainX, trainX[0:1000])
	kernel_test = arc_cosine(testX, trainX[0:1000])

	trainX_kpca = kpca.transform(kernel_train)
	testX_kpca = kpca.transform(kernel_test)
	print testX_kpca.shape

	#fit the svm model and compute accuaracy measure
	clf = svm.SVC(kernel=arc_cosine)
	clf.fit(trainX_kpca, trainY)

	pred = clf.predict(testX_kpca)
	print accuracy_score(testY, pred)
	print('total : %d, correct : %d, incorrect : %d\n' %(len(pred), np.sum(pred == testY), np.sum(pred != testY)))

	print('Test Time : %f Minutes\n' %((time.time()-start)/60))

if __name__ == '__main__':
		main()	