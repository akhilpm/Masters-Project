'''
KPCA based feature engineering for 20-newsgroup document classification
Author : Akhil P M
Kernel used : Arc-cosine Kernel
'''

from settings import *
import kernel

def size_mb(docs):
	return sum(len(s.encode('utf-8')) for s in docs) / 1e6


def trim(s):
	"""Trim string to fit on terminal (assuming 80-column display)"""
	return s if len(s) <= 80 else s[:77] + "..."


def get20newsgroup_data(op):

	#set the timer
	start = time()

	#ignore all warnings
	warnings.filterwarnings("ignore")

	# Display progress logs on stdout
	logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


	(opts, args) = op.parse_args()
	#print args, opts
	#if len(args) > 0:
		#op.error("this script takes no arguments")
		#sys.exit(1)

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

	return trainX, testX, trainY, testY

