'''
KPCA based feature engineering for 20 news-group text data classification.
Author : Akhil P M
Kernel used : Arc-cosine Kernel
'''

from settings import *
import kernel
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
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
	

	norm_matrix = np.outer(temp1,temp2) #the matix of k_xx, and K_yy's
	theta = np.arccos( np.maximum( np.minimum(Mat/np.sqrt(norm_matrix), 1.0), -1.0))
	M = np.multiply(np.power(norm_matrix, n_l/2.0), compute_J(n_l, theta)) / np.pi

	return M
'''
def arc_cosine_vector(X, Y, n_l, Mat, temp1, temp2):
	lenX = X.shape[0]
	incr = 1000
	M = np.zeros((lenX, Y.shape[0]))
	for i in range(0,lenX,incr):
		M[i:i+incr] = arc_cosine(n_l, Mat, temp1, temp2)

	return M	
'''

def size_mb(docs):
	return sum(len(s.encode('utf-8')) for s in docs) / 1e6


def trim(s):
	"""Trim string to fit on terminal (assuming 80-column display)"""
	return s if len(s) <= 80 else s[:77] + "..."


names = ["Linear SVM", "Decision Tree", "Random Forest",
		"AdaBoost Classifier","Logistic Regression"]


classifiers = [
	SVC(kernel="linear", C=3.4),
	DecisionTreeClassifier(),
	RandomForestClassifier(n_estimators=300, n_jobs=-1),
	AdaBoostClassifier(n_estimators=70),
	LogisticRegression(random_state=1, C=0.4)]

def main():

	#set the timer
	start = time()

	#ignore all warnings
	warnings.filterwarnings("ignore")

	# Display progress logs on stdout
	logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


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

	(opts, args) = op.parse_args()
	if len(args) > 0:
		op.error("this script takes no arguments.")
		sys.exit(1)

	# Load some categories from the training set
	if opts.all_categories:
		categories = None
	else:
		categories = [
			'alt.atheism',
			'talk.religion.misc',
			'comp.graphics',
			'sci.space',
		]

	if opts.filtered:
		remove = ('headers', 'footers', 'quotes')
	else:
		remove = ()

	print("Loading 20 newsgroups dataset for categories:")
	print(categories if categories else "all")

	data_train = fetch_20newsgroups(subset='train', categories=categories,
		shuffle=True, random_state=42, remove=remove)

	data_test = fetch_20newsgroups(subset='test', categories=categories,
		shuffle=True, random_state=42, remove=remove)

	print('\n!!! Data Loading Completed !!!\n')


	categories = data_train.target_names    # for case categories == None

	data_train_size_mb = size_mb(data_train.data)
	data_test_size_mb = size_mb(data_test.data)

	print("%d documents - %0.3fMB (training set)" % (len(data_train.data), data_train_size_mb))
	print("%d documents - %0.3fMB (test set)" % (len(data_test.data), data_test_size_mb))
	print("%d categories" % len(categories))
	print('\n'+'=' * 80+'\n')

	# split a training set and a test set
	trainY, testY = data_train.target, data_test.target

	print("Extracting features from the training data using a sparse vectorizer")
	t0 = time()
	if opts.use_hashing:
		vectorizer = HashingVectorizer(stop_words='english', non_negative=True, 
			n_features=opts.n_features)
		trainX = vectorizer.transform(data_train.data)
	else:
		vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
		trainX = vectorizer.fit_transform(data_train.data)

	duration = time() - t0
	print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
	print("n_samples: %d, n_features: %d" % trainX.shape)
	print('\n'+'=' * 80+'\n')

	print("Extracting features from the test data using the same vectorizer")
	t0 = time()
	testX = vectorizer.transform(data_test.data)
	duration = time() - t0
	print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
	print("n_samples: %d, n_features: %d" % testX.shape)
	print('\n'+'=' * 80+'\n')

	# mapping from integer feature name to original token string
	if opts.use_hashing:
		feature_names = None
	else:
		feature_names = vectorizer.get_feature_names()

	if opts.select_chi2:
		print("Extracting %d best features by a chi-squared test" %opts.select_chi2)
		t0 = time()
		ch2 = SelectKBest(chi2, k=opts.select_chi2)
		trainX = ch2.fit_transform(trainX, trainY)
		testX = ch2.transform(testX)
		if feature_names:
			# keep selected feature names
			feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
		print("done in %fs" % (time() - t0))
		print()

	if feature_names:
		feature_names = np.asarray(feature_names)

	'''
	trainX = np.load('trainX.npy')
	testX = np.load('testX.npy')
	trainY = np.load('trainY.npy')
	testY = np.load('testY.npy')
	'''

	#shuffle the training data
	shuffle = np.random.permutation(trainX.shape[0])
	trainX = trainX[shuffle]
	trainY = trainY[shuffle]

	"""param = a vector of n(degree) values at each layer """
	param = np.array([0, 2, 2])
	print param
	no_of_layers = len(param)

	'''
	for name, clf in zip(names, classifiers):
		clf.fit(trainX, trainY)
		pred = clf.predict(testX)
		print('classifier : %s, Accuracy :%f%% ' %(name, accuracy_score(testY, pred)*100))
		print('total : %d, correct : %d, incorrect : %d\n' %(len(pred), np.sum(pred == testY), np.sum(pred != testY)))
	
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
	temp1 = kpcaX.dot(kpcaX.T).diagonal()
	#temp1 = np.diag(np.dot(kpcaX, kpcaX.T))
	Mat1 = kpcaX.dot(kpcaX.T).todense()

	temp2 = trainX.dot(trainX.T).diagonal()
	Mat2 = trainX.dot(kpcaX.T).todense()

	temp3 = testX.dot(testX.T).diagonal()
	Mat3 = testX.dot(kpcaX.T).todense()

	# Univariate feature selection with F-test for feature scoring
	# We use the default selection function: the 10% most significant features
	selector = SelectPercentile(f_classif, percentile=5)
	#selector = SelectKBest(chi2, k=200)

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
	np.save('trainX_feat', trainX)
	np.save('testX_feat', testX)
	np.save('trainY_feat', trainY)
	np.save('testY_feat', testY)

	#fit the svm model and compute accuaracy measure
	#parameters = {'n_neighbors' : list(np.arange(20)+1)}
	#clf = GridSearchCV(KNeighborsClassifier(weights='distance', n_jobs=-1), parameters)
	#clf = svm.SVC(kernel='linear')
	#clf = svm.SVC(kernel=kernel.arc_cosine, cache_size=2048)
	#clf.fit(trainX, trainY)

	for name, clf in zip(names, classifiers):
		clf.fit(trainX, trainY)
		pred = clf.predict(testX)
		print('classifier : %s, Accuracy :%f%% ' %(name, accuracy_score(testY, pred)*100))
		print('total : %d, correct : %d, incorrect : %d\n' %(len(pred), np.sum(pred == testY), np.sum(pred != testY)))

	pred = clf.predict(testX)
	print accuracy_score(testY, pred)
	print('total : %d, correct : %d, incorrect : %d\n' %(len(pred), np.sum(pred == testY), np.sum(pred != testY)))

	print('Test Time : %f Minutes\n' %((time()-start)/60))


if __name__ == '__main__':
	main()

