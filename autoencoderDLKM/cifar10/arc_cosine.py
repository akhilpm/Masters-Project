'''
Multi-layer arc-cosine: Vectorized version
Note: The dot product of matrices in the kernel computation will eat up so much RAM
      reducing the precision of float data is a nice option. A decomposed version
      of the same is coming soon.
Author: Akhil P M
'''

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
import gc

from settings import *
import encoderRGB

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
	param = np.array([1,1,1,1,1,1,1])
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
	incr = 2000
	M = np.zeros((lenX, Y.shape[0]))
	for i in range(0,lenX,incr):
		M[i:i+incr] = arc_cosine_vector(X[i:i+incr], Y)

	return M	



def main():

	start = time.time()
	#'''
	W, b = encoderRGB.execute_sparse_autoencoder()

	trainX = np.zeros((30000, 3072))
	trainY = np.zeros(30000)
	testX = np.zeros((10000, 3072))
	testY = np.zeros(10000)

	fo = open('data_batch_1', 'rb')
	dict = cPickle.load(fo)
	fo.close()
	data = dict['data']
	label = dict['labels']
	label = np.asarray(label)
	trainX[0:10000] = data
	trainY[0:10000] = label

	fo = open('data_batch_2', 'rb')
	dict = cPickle.load(fo)
	fo.close()
	data = dict['data']
	label = dict['labels']
	label = np.asarray(label)
	trainX[10000:20000] = data
	trainY[10000:20000] = label
	
	fo = open('data_batch_3', 'rb')
	dict = cPickle.load(fo)
	fo.close()
	data = dict['data']
	label = dict['labels']
	label = np.asarray(label)
	trainX[20000:] = data
	trainY[20000:] = label

	fo = open('test_batch', 'rb')
	dict = cPickle.load(fo)
	fo.close()
	data = dict['data']
	label = dict['labels']
	label = np.asarray(label)
	testX[0:] = data
	testY[0:] = label


	trainX = trainX/255.0
	testX = testX/255.0


	trainX = np.transpose(trainX)
	testX = np.transpose(testX)
	trainX = encoderRGB.extract_feature(W,b, trainX)
	testX = encoderRGB.extract_feature(W,b, testX)

	trainX = np.transpose(trainX)
	testX = np.transpose(testX)

	trainX = trainX.astype(np.float32)
	testX = testX.astype(np.float32)

	'''
	trainX = np.load('trainX.npy')
	testX = np.load('testX.npy')
	trainY = np.load('trainY.npy')
	testY = np.load('testY.npy')
	'''
	np.save('trainX', trainX)
	np.save('testX', testX)
	np.save('trainY', trainY)
	np.save('testY', testY)
	#'''
	#sss = StratifiedShuffleSplit(mnist.target, 1, test_size=0.1, train_size=0.1, random_state=0)
	#for train_index, test_index in sss:
	#	trainX, testX = mnist.data[train_index], mnist.data[test_index]
	#	trainY, testY = mnist.target[train_index], mnist.target[test_index]


	clf = svm.SVC(kernel=arc_cosine, cache_size=4096)
	#clf = svm.SVC(kernel = 'poly') #gaussian kernel is used
	clf.fit(trainX, trainY)

	pred = clf.predict(testX)
	print accuracy_score(testY, pred)
	print('total : %d, correct : %d, incorrect : %d\n' %(len(pred), np.sum(pred == testY), np.sum(pred != testY)))

	print('Test Time : %f Minutes\n' %((time.time()-start)/60))

	#pred = clf.predict(trainX)
	#print accuracy_score(trainY, pred)
	#print('total : %d, correct : %d, incorrect : %d\n' %(len(pred), np.sum(pred == trainY), np.sum(pred != trainY)))

	print('Execution Time : %f Minutes\n' %((time.time()-start)/60))



if __name__ == '__main__':
	main()	
