#!/usr/bin/env python -W ignore::DeprecationWarning
'''

KPCA based feature engineering for 20-newsgroup document classification with
combination of kernels in each layers
Author : Akhil P M
Kernel used : Arc-cosine Kernel, Gaussian Kernel, Polynomial kernel

'''

import kernel
from settings import *
from umkl_new import *
from getdata_20NG import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


n=3000
n_kernels = 4
D = np.zeros((n,n))
M = np.zeros((n,n))
P = np.zeros((n,n))

matP = np.zeros((n_kernels, n_kernels))
vecQ = np.zeros((n_kernels,1))
gamma = 0.01

names = ["Linear SVM", "Decision Tree", "Random Forest",
		"AdaBoost Classifier", "Logistic Regression"]


classifiers = [
	SVC(kernel="linear", C=3.4,gamma=0.1),
	DecisionTreeClassifier(),
	RandomForestClassifier(n_estimators=300, n_jobs=-1),
	AdaBoostClassifier(n_estimators=70),
	LogisticRegression(random_state=1, C=0.4)]


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
	param = np.array([0, 3, 3])
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


def stratified_sampling(trainX, trainY):
	""" stratified inputs for KPCA is extracted by this function """

	no_of_classes = len(np.unique(trainY))
	representers = np.zeros(no_of_classes)
	no_of_reps = int(3000/no_of_classes)
	kpcaX = np.zeros((3000, trainX.shape[1]))
	count = 0
	index = 0

	for i in xrange(trainX.shape[0]):
		label = trainY[i]
		if representers[label] < no_of_reps:
			kpcaX[index] = trainX[i]
			index += 1
		representers[label] += 1

		if representers[label] == no_of_reps:
			count += 1
		if count == no_of_classes:
			break

	return kpcaX




def uncertainity_sampling(trainX, trainY):
	""" sample most uncertain points using active learning techniques,
	specifically using label propagation algorithm """

	n_total_samples = len(trainY)
	n_labeled_points = 100
	unlabelled_indices = np.arange(n_total_samples)[n_labeled_points:]

	y_train = np.copy(trainY)
	y_train[unlabelled_indices] = -1

	lp_model = label_propagation.LabelSpreading(kernel='knn', alpha=1.0)
	lp_model.fit(trainX, y_train)

	# compute the entropies of transduced label distributions
	pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)

	# select 3000 digit examples that the classifier is most uncertain about
	uncertainty_index = np.argsort(pred_entropies)[-3000:]
	print(uncertainty_index)

	kpcaX = trainx[uncertainty_index]
	print(kpcaX.shape)

	return kpcaX




def multi_KPCA(trainX, trainY, testX, testY, param, k_type, layer):
	""" KPCA using combination of kernels """

	kpca = KernelPCA(kernel='precomputed')
	#kpcaX = stratified_sampling(trainX, trainY) #trainX[0:3000]
	kpcaX = trainX[0:3000]
	kpcaY = trainY[0:3000]

	#for i in range(10):
	#	print np.sum(kpcaY==i),

	kpca_train = np.zeros((3000, 3000))
	kernel_train = np.zeros((trainX.shape[0], 3000))
	kernel_test = np.zeros((testX.shape[0], 3000))

	#get the coefficients
	mu = getUMKL_coefficients(trainX[:n], k_type, param, layer)
	print(mu)

	kpca_train = getUMKL_gram_matrix(kpcaX, kpcaX, k_type, param, mu, layer)
	kernel_train = getUMKL_gram_matrix(trainX, kpcaX, k_type, param, mu, layer)
	kernel_test = getUMKL_gram_matrix(testX, kpcaX, k_type, param, mu, layer)

	kpca.fit(kpca_train)
	trainX_kpca = kpca.transform(kernel_train)
	testX_kpca = kpca.transform(kernel_test)

	gc.collect()
	get_individual_kernel_performance(kpcaX, trainX, trainY, testX, testY, k_type, param, mu, layer)
	return trainX_kpca, testX_kpca




def read_cmd_arguments(no_of_layers, no_of_kernels):
	""" get parameters of each layer as cmd arguments"""

	config = sys.argv[1]
	param = genfromtxt(config, delimiter=',')
	print(param)

	k_type = genfromtxt('kernels.csv', delimiter=',')

	return param, k_type



def main():

	#ignore all warnings
	warnings.filterwarnings("ignore")

	#set the parameters
	no_of_layers = 4
	no_of_kernels = 5
	kparam = np.array([0,3,3])

	""" param = a vector of kernel parameter values at each layer """
	param, k_type = read_cmd_arguments(no_of_layers, no_of_kernels)

		# parse commandline arguments
	op = OptionParser()
	op.add_option("--report",action="store_true", dest="print_report",
		help="Print a detailed classification report.")
	op.add_option("--chi2_select", action="store", type="int", dest="select_chi2",
		help="Select some number of features using a chi-squared test")
	op.add_option("--top10",action="store_true", dest="print_top10", 
		help="Print ten most discriminative terms per class"" for every classifier.")
	op.add_option("--all_categories",action="store_true", dest="all_categories", 
		help="Whether to use all categories or not.")
	op.add_option("--use_hashing", action="store_true", 
		help="Use a hashing vectorizer.")
	op.add_option("--n_features", action="store", type=int, default=2 ** 16, 
		help="n_features when using the hashing vectorizer.")
	op.add_option("--filtered", action="store_true", 
		help="Remove newsgroup information that easily overfits: ""headers, signatures, and quoting.")

	#set the timer
	start = time()

	#get the data after preprocessing
	trainX, testX, trainY, testY = get20newsgroup_data(op)
	print('\n!!! Data Loading Completed !!!\n')	

	#shuffle the training data
	shuffle = np.random.permutation(trainX.shape[0])
	trainX = trainX[shuffle]
	trainY = trainY[shuffle]
	

	selector = SelectPercentile(f_classif, percentile=5)

	#extract the features using KPCA
	for i in xrange(no_of_layers):
	
		trainX_kpca, testX_kpca = multi_KPCA(trainX, trainY, testX, testY, param[i], k_type[i], i+1)
	

		selector.fit(trainX_kpca, trainY)
		trainX = selector.transform(trainX_kpca)
		testX = selector.transform(testX_kpca)
		clf = SVC(kernel="linear", C=3.4)
		clf.fit(trainX, trainY)
		pred = clf.predict(testX)
		print(accuracy_score(testY, pred))

		print(trainX_kpca.shape)
		print(trainX.shape)
		print('============================ Layer %d Completed ============================' %(i+1))

	print(testX.shape)

	#save the new featurset for further exploration
	np.save('trainX_feat', trainX)
	np.save('testX_feat', testX)
	np.save('trainY_feat', trainY)
	np.save('testY_feat', testY)

	#fit the svm model and compute accuaracy measure
	parameters = {'n_neighbors' : list(np.arange(20)+1)}
	#clf = GridSearchCV(KNeighborsClassifier(weights='distance', n_jobs=-1), parameters)
	#clf = svm.SVC(kernel=arc_cosine, cache_size=2048)
	#clf.fit(trainX, trainY)

	for name, clf in zip(names, classifiers):
		clf.fit(trainX, trainY)
		pred = clf.predict(testX)
		print('classifier : %s, Accuracy :%f%% ' %(name, accuracy_score(testY, pred)*100))
		print('total : %d, correct : %d, incorrect : %d\n' %(len(pred), np.sum(pred == testY), np.sum(pred != testY)))


	#pred = clf.predict(testX)
	#print(accuracy_score(testY, pred))
	#print(confusion_matrix(testY, pred))
	#print(clf.best_params_)
	#print('total : %d, correct : %d, incorrect : %d\n' %(len(pred), np.sum(pred == testY), np.sum(pred != testY)))

	print('Test Time : %f Minutes\n' %((time()-start)/60))
	print('completed time ' + str(datetime.now().hour) + ':' + str(datetime.now().minute))


if __name__ == '__main__':
	main()

