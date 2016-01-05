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
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


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


def arc_cosine(n_l, Mat, temp1, temp2):
	
	#M = np.dot(X, Y.T)
	#temp1 = np.diag(np.dot(X, X.T))
	#temp2 = np.diag(np.dot(Y, Y.T))	

	norm_matrix = np.outer(temp1,temp2) #the matix of k_xx, and K_yy's
	theta = np.arccos( np.maximum( np.minimum(Mat/np.sqrt(norm_matrix), 1.0), -1.0))
	M = np.multiply(np.power(norm_matrix, n_l/2.0), compute_J(n_l, theta)) / np.pi

	return M

def arc_cosine_vector(X, Y, n_l, Mat, temp1, temp2):
	lenX = X.shape[0]
	incr = 1000
	M = np.zeros((lenX, Y.shape[0]))
	for i in range(0,lenX,incr):
		M[i:i+incr] = arc_cosine_vector(n_l, Mat, temp1, temp2)

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
	param = np.array([0, 0, 1])
	no_of_layers = len(param)

	
	'''
	#Initial feature selection
	forest = ExtraTreesClassifier(n_estimators=400, random_state=0, n_jobs=-1)
	forest.fit(trainX, trainY)
	print forest.n_features_
	importances = forest.feature_importances_
	indices = np.argsort(importances)[::-1]
	trainX = select_features(trainX, importances, indices)
	print trainX.shape
	testX = select_features(testX, importances, indices)
	'''

	#extract the features using KPCA
	kpca = KernelPCA(kernel='precomputed')
	kpcaX = trainX[0:3000]

	#all the temp variables needed in the subsequent stages are pre-computed
	temp1 = np.diag(np.dot(kpcaX, kpcaX.T))
	Mat1 = np.dot(kpcaX, kpcaX.T)

	temp2 = np.diag(np.dot(trainX, trainX.T))
	Mat2 = np.dot(trainX, kpcaX.T)

	temp3 = np.diag(np.dot(testX, testX.T))
	Mat3 = np.dot(testX, kpcaX.T)


	for i in xrange(len(param)):
		n_l = param[i]
		print('complutation for layer %d\n' %(i+1))
		kpca_train = arc_cosine(param[i], Mat1, temp1, temp1)
		kpca.fit(kpca_train)

		kernel_train = arc_cosine(param[i], Mat2, temp2, temp1)
		kernel_test = arc_cosine(param[i], Mat3, temp3, temp1)

		trainX_kpca = kpca.transform(kernel_train)
		testX_kpca = kpca.transform(kernel_test)


		#fit the random forest model to evaluate featues
		forest = ExtraTreesClassifier(n_estimators=400, random_state=0, n_jobs=-1)
		#forest = RandomForestClassifier(n_estimators=300, n_jobs=-1)
		#forest = DecisionTreeClassifier()
		forest.fit(trainX_kpca, trainY)
		print forest.n_features_

		importances = forest.feature_importances_
		indices = np.argsort(importances)[::-1]
		#for f in range(kpcaX.shape[1]):
		#	print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
			#chumma = raw_input('### wait for key press ### ')

		print trainX_kpca.shape
		trainX = select_features(trainX_kpca, importances, indices)
		print trainX.shape
		testX = select_features(testX_kpca, importances, indices)
		features_selected = trainX.shape[1]
		kpcaX = trainX[0:3000]

		if i < no_of_layers-1:
			zeros1 = np.zeros(len(temp1))
			temp1 = np.multiply(np.power(temp1, n_l), compute_J(n_l, zeros1)) / np.pi
			Mat1 = np.copy(kpca_train)

			zeros2 = np.zeros(len(temp2))
			temp2 = np.multiply(np.power(temp2, n_l), compute_J(n_l, zeros2)) / np.pi
			Mat2 = np.copy(kernel_train)

			zeros3 = np.zeros(len(temp3))
			temp3 = np.multiply(np.power(temp3, n_l), compute_J(n_l, zeros3)) / np.pi
			Mat3 = np.copy(kernel_test)						


	print testX.shape

	#save the new featurset for further exploration
	#np.save('trainX', trainX)
	#np.save('testX', testX)
	#np.save('trainY', trainY)
	#np.save('testY', testY)

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

