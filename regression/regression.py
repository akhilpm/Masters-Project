'''
KPCA based feature engineering for MNIST-RAND handwritten digits classification
Author : Akhil P M
Kernel used : Arc-cosine Kernel
'''

from settings import *
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
#from sklearn.feature_selection import SelectKBest, chi2


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

	boston = datasets.load_boston()
	boston.data = preprocessing.scale(boston.data)

	trainX, testX, trainY, testY = train_test_split(boston.data, boston.target, test_size = 0.3, random_state=42)

	print('\n!!! Data Loading Completed !!!\n')

	#shuffle the training data
	#shuffle = np.random.permutation(trainX.shape[0])
	#trainX = trainX[shuffle]
	#trainY = trainY[shuffle]

	"""param = a vector of n(degree) values at each layer """
	param = np.array([2, 0])
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
	kpcaX = trainX[0:300]

	#all the temp variables needed in the subsequent stages are pre-computed
	temp1 = np.diag(np.dot(kpcaX, kpcaX.T))
	Mat1 = np.dot(kpcaX, kpcaX.T)

	temp2 = np.diag(np.dot(trainX, trainX.T))
	Mat2 = np.dot(trainX, kpcaX.T)

	temp3 = np.diag(np.dot(testX, testX.T))
	Mat3 = np.dot(testX, kpcaX.T)

	# Univariate feature selection with F-test for feature scoring
	# We use the default selection function: the 10% most significant features
	#selector = SelectPercentile(f_classif, percentile=5)
	selector = SelectFromModel(LassoCV(), threshold=.5)

	for i in xrange(len(param)):
		n_l = param[i]
		print('computation for layer %d\n' %(i+1))
		kpca_train = arc_cosine(param[i], Mat1, temp1, temp1)
		kpca.fit(kpca_train)

		kernel_train = arc_cosine(param[i], Mat2, temp2, temp1)
		kernel_test = arc_cosine(param[i], Mat3, temp3, temp1)

		trainX_kpca = kpca.transform(kernel_train)
		testX_kpca = kpca.transform(kernel_test)

	
		selector.fit(trainX_kpca, trainY)

		print trainX_kpca.shape
		trainX = selector.transform(trainX_kpca)
		print trainX.shape
		testX = selector.transform(testX_kpca)
		kpcaX = trainX[0:300]

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


	print testX.shape, '\n'

	#save the new featurset for further exploration
	np.save('trainX_feat', trainX)
	np.save('testX_feat', testX)
	np.save('trainY_feat', trainY)
	np.save('testY_feat', testY)
	
	#fit the svm model and compute accuaracy measure
	#clf = svm.SVC(kernel=kernel.arc_cosine, cache_size=2048)
	#regr = SVR(kernel='rbf', C=1e3, gamma=0.1)
	regr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5, n_jobs=-1,
		param_grid={"C": np.logspace(-2, 2, 20),"gamma": np.logspace(-2, 2, 20)})
	#[1e0, 1e1, 1e2, 1e3]
	#regr = SVR(kernel='linear', C=1e3)
	#regr = SVR(kernel='poly', C=1e3, degree=2)
	regr.fit(trainX, trainY)

	pred = regr.predict(testX)
	print("Mean Square Error(MSE): %.2f" % MSE(pred, testY))
	print('Variance score: %.2f' % regr.score(testX, testY))
	print('R2 score: %.2f\n' % r2_score(pred, testY))

	pred = regr.predict(trainX)
	print("Mean Square Error(MSE): %.2f" % MSE(pred, trainY))
	print('Variance score: %.2f' % regr.score(trainX, trainY))
	print('R2 score: %.2f' % r2_score(pred, trainY))

	print('Test Time : %f Minutes\n' %((time.time()-start)/60))


if __name__ == '__main__':
	main()

