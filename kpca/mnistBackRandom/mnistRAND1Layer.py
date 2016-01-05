'''
KPCA based feature engineering for MNIST-RAND handwritten digits classification
Author : Akhil P M
Kernel used : Arc-cosine Kernel
'''

from settings import *
from feat_select import *
import kernel
from metric_learn import LMNN
from sklearn.neighbors import KNeighborsClassifier


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



def main():

	#set the timer
	start = time.time()

	#load the data
	'''
	trainX = genfromtxt('mnist_train.csv', delimiter=',')
	testX = genfromtxt('mnist_test.csv', delimiter=',')
	print('\n!!! Data Loading Completed !!!\n')

	#separate X and Y values for training
	n = trainX.shape[1]-1
	trainY = trainX[:,n]
	trainY = trainY.astype(np.int32)
	trainX = np.delete(trainX, n, -1)

	#separate X and Y values for testing
	testY = testX[:,n]
	testY = testY.astype(np.int32)
	testX = np.delete(testX, n, -1)


	#scale down features to the range [0, 1]
	trainX = trainX/255.0
	trainX = trainX.astype(np.float32)
	testX = testX/255.0
	testX = testX.astype(np.float32)

	np.save('trainX', trainX)
	np.save('testX', testX)
	np.save('trainY', trainY)
	np.save('testY', testY)
	'''
	trainX = np.load('trainX.npy')
	testX = np.load('testX.npy')
	trainY = np.load('trainY.npy')
	testY = np.load('testY.npy')
	print('\n!!! Data Loading Completed !!!\n')	

	#shuffle the training data
	shuffle = np.random.permutation(trainX.shape[0])
	trainX = trainX[shuffle]
	trainY = trainY[shuffle]

	"""param = a vector of n(degree) values at each layer """
	param = np.array([0])

	#extract the features using KPCA
	kpca = KernelPCA(kernel='precomputed')
	kpcaX = trainX[0:3000]

	for i in xrange(len(param)):
		kpca_train = arc_cosine(kpcaX, kpcaX, param[i])
		kpca.fit(kpca_train)

		kernel_train = arc_cosine(trainX, kpcaX, param[i])
		kernel_test = arc_cosine(testX, kpcaX, param[i])

		trainX_kpca = kpca.transform(kernel_train)
		testX_kpca = kpca.transform(kernel_test)


		#fit the random forest model to evaluate featues
		forest = ExtraTreesClassifier(n_estimators=400, random_state=0, n_jobs=-1)
		forest.fit(trainX_kpca, trainY)

		importances = forest.feature_importances_
		indices = np.argsort(importances)[::-1]
		print len(indices)
		#for f in range(kpcaX.shape[1]):
		#	print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
			#chumma = raw_input('### wait for key press ### ')

		print trainX_kpca.shape
		trainX = select_features(trainX_kpca, importances, indices)
		print trainX.shape
		testX = select_features(testX_kpca, importances, indices)
		kpcaX = trainX[0:3000]

	print testX.shape

	#save the new featurset for further exploration
	np.save('trainX_feat', trainX)
	np.save('testX_feat', testX)
	np.save('trainY_feat', trainY)
	np.save('testY_feat', testY)

	#fit the svm model and compute accuaracy measure
	clf = KNeighborsClassifier(n_neighbors=15, weights='distance')
	#clf = svm.SVC(kernel='linear')
	clf.fit(trainX, trainY)

	pred = clf.predict(testX)
	print accuracy_score(testY, pred)
	print('total : %d, correct : %d, incorrect : %d\n' %(len(pred), np.sum(pred == testY), np.sum(pred != testY)))

	print('Test Time : %f Minutes\n' %((time.time()-start)/60))


if __name__ == '__main__':
	main()

