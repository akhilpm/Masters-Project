'''
Softmax regression on MNIST dataset 
Author: Akhil P M
'''

from sklearn.datasets.mldata import fetch_mldata
import numpy as np
from sklearn.cross_validation import train_test_split
import numexpr as ne #multiprocessing accelator for numpy operations, uses less memory
import scipy
import scipy.optimize
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import gc
import time

def costFunction(theta,trainX,trainY):
	global lambdaa, ntrain, num_classes
	m,n = trainX.shape

	#take all theta(k) values as a vector to compute numerator in P(yi=k/xi;theta)
	#by a vectorized implementation
	theta = theta.reshape((num_classes, ntrain))
	theta_vec = theta[trainY-1]
	theta_dash = np.transpose(theta)
	temp = trainX*theta_vec
	temp = ne.evaluate('sum(temp,axis=1)')
	temp = ne.evaluate('exp(temp)') #parallelized
	hyp_sum = np.zeros(m)
	for i in np.arange(m):
		hyp_sum[i] = np.sum(np.exp(np.dot(trainX[i],theta_dash)))

	#calculate the hypothesis value
	#gc.collect()
	temp /= hyp_sum
	hyp = ne.evaluate('log(temp)')
	jval = -ne.evaluate('sum(hyp)')/m + lambdaa/2*ne.evaluate('sum(theta*theta)')

	#compute the gradient wrt to all theta_k
	grad = np.random.random_sample((num_classes,n))
	for k in xrange(num_classes):
		gradK = 1*(trainY==k+1)
		pos = np.ones(m,dtype=int)*k
		theta_vec = theta[pos]
		temp = trainX*theta_vec
		temp = ne.evaluate('sum(temp,axis=1)')
		temp = ne.evaluate('exp(temp)')
		hyp = temp/hyp_sum
		gradK = gradK-hyp
		gradK = gradK.reshape((m, 1))
		grad[k] = -np.sum(trainX*(gradK),axis=0)/m
		grad[k] += lambdaa*theta[k]
	#convert grad into a vector	
	grad = grad.flatten()	
	return jval,grad			


def main():
	
	global lambdaa, ntrain, num_classes 
	start = time.time()
	lambdaa = 0.00001  
	max_iterations = 500

	#Load the Digit Data Set
	mnist = fetch_mldata('MNIST original')
	#min_max_scaler = preprocessing.MinMaxScaler()
	#mnist.data = min_max_scaler.fit_transform(mnist.data)
	mnist.data = mnist.data/255.0
	mnist.target = mnist.target.astype(np.int32)
	seed = np.random.randint(1,30000)
	rand = np.random.RandomState(seed)
	items = len(mnist.target)
	indices = rand.randint(items, size = 70000)
	trindex = indices[0:50000]
	tsindex = indices[50000:]

	trainX = mnist.data[trindex]
	testX = mnist.data[tsindex]
	trainY = mnist.target[trindex]
	testY = mnist.target[tsindex]

	#trainX,testX,trainY,testY = train_test_split(mnist.data,mnist.target,test_size=0.3)
	mtrain,ntrain = trainX.shape
	mtest,ntest = testX.shape

	#Append one to the first column of the training data
	ones = np.ones((mtrain,1), dtype=int)
	trainX = np.append(ones,trainX,axis=1)
	ones = np.ones((mtest,1), dtype=int)
	testX = np.append(ones,testX,axis=1)
	ntrain = ntest = ntrain+1

	# make digits range as 1-10
	trainY = trainY+1 
	testY = testY+1
	num_classes = len(np.unique(trainY))
	theta = np.random.random_sample((num_classes, ntrain)).flatten()

	"""do the optimization using L-BFGS algoritm"""
	result = scipy.optimize.minimize(costFunction,
					theta, args=(trainX,trainY,), method='L-BFGS-B', 
                    jac=True, options={'maxiter': max_iterations, 'disp' : True})

	theta = result.x.reshape((num_classes, ntrain))
	theta_dash = np.transpose(theta)


	"""" classify the test datapoints using the learned parameters"""
	pred = np.ones(mtest, dtype=int)
	for i in xrange(mtest):
		temp = np.exp(np.dot(testX[i], theta_dash))
		pred[i] = np.argmax(temp)+1

	print accuracy_score(testY, pred)
	print('total : %d, correct : %d, incorrect : %d\n' %(len(pred), np.sum(pred==testY), np.sum(pred!=testY)))

	print('execution time(in seconds):%f\n' %(time.time()-start))

if __name__ == '__main__':
	main()	