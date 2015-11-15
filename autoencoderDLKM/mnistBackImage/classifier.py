'''
Softmax regression classifier on MNIST dataset 
Author: Akhil P M
'''

from settings import *
import mnistImageEncode

def costFunction(theta,trainX,trainY):
	global lambdaa, ntrain, num_classes
	m,n = trainX.shape

	#take all theta(k) values as a vector to compute numerator in P(yi=k/xi;theta)
	#by a vectorized implementation
	theta = theta.reshape((num_classes, ntrain))
	theta_vec = theta[trainY-1]
	theta_dash = np.transpose(theta)
	temp = trainX*theta_vec
	temp = np.sum(temp,axis=1)
	temp = np.exp(temp) #parallelized
	hyp_sum = np.zeros(m)
	for i in np.arange(m):
		hyp_sum[i] = np.sum(np.exp(np.dot(trainX[i],theta_dash)))

	#calculate the hypothesis value
	#gc.collect()
	temp /= hyp_sum
	hyp = np.log(temp)
	jval = -np.sum(hyp)/m + lambdaa/2*np.sum(theta*theta)

	#compute the gradient wrt to all theta_k
	grad = np.random.random_sample((num_classes,n))
	for k in xrange(num_classes):
		gradK = 1*(trainY==k+1)
		pos = np.ones(m,dtype=int)*k
		theta_vec = theta[pos]
		temp = trainX*theta_vec
		temp = np.sum(temp,axis=1)
		temp = np.exp(temp)
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
	max_iterations = 400

	#Load the Digit Data Set
	trainX = genfromtxt('mnist_train.csv', delimiter=',')
	testX = genfromtxt('mnist_test.csv', delimiter=',')
	print('\n!!! Data Loading Completed !!!\n')

	n = trainX.shape[1]-1	
	trainY = trainX[:,n]
	trainY = trainY.astype(np.int32)
	trainX = np.delete(trainX, n, -1)

	testY = testX[:,n]
	testY = testY.astype(np.int32)
	testX = np.delete(testX, n, -1)

	trainX = trainX.astype(np.float32)
	testX = testX.astype(np.float32)

	W, b = mnistImageEncode.execute_sparse_autoencoder(trainX)
	

	""" Get the features from the learned autoencoder """
	#W = np.transpose(W)
	trainX = np.transpose(trainX)
	testX = np.transpose(testX)
	trainX = mnistImageEncode.extract_feature(W,b, trainX)
	testX = mnistImageEncode.extract_feature(W,b, testX)

	trainX = np.transpose(trainX)
	testX = np.transpose(testX)
	#print trainX.shape
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

	print('execution time(in Minutes):%f\n' %((time.time()-start)/60))

if __name__ == '__main__':
	main()	
