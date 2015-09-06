import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
#from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc


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


def arc_cosine(X, Y):
	"""param = a vector of n(degree) values at each layer """
	param = np.array([0,1,1])
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

#Data = np.array([[1,2,3], [1,-1,1], [2,1,1]])
#y = Data
#M = arc_cosine(Data,y)

#digits = datasets.load_digits()
#X = digits.data
#Y = digits.target

#trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.3)
#trainX, trainY = load_svmlight_file("/home/akhi/Documents/miniproject/egMulticlass/data/train.dat")
#testX, testY = load_svmlight_file("/home/akhi/Documents/miniproject/egMulticlass/data/test.dat")
trainX = np.loadtxt('/home/akhi/Documents/miniproject/egMulticlass/data/trainData.dat')
testX = np.loadtxt('/home/akhi/Documents/miniproject/egMulticlass/data/testData.dat')
trainY = np.loadtxt('/home/akhi/Documents/miniproject/egMulticlass/data/trainLabel.dat')
testY = np.loadtxt('/home/akhi/Documents/miniproject/egMulticlass/data/testLabel.dat')


#clf = svm.SVC(kernel=arc_cosine)
clf = svm.SVC(kernel=arc_cosine, probability=True)
y_score = clf.fit(trainX, trainY).decision_function(testX)
#clf = svm.SVC(kernel = 'poly') #gaussian kernel is used
#clf.fit(trainX, trainY)

pred = clf.predict(testX)
print accuracy_score(testY, pred)
print('total : %d, correct : %d, incorrect : %d\n' %(len(pred), np.sum(pred == testY), np.sum(pred != testY)))

pred = clf.predict(trainX)
print accuracy_score(trainY, pred)
print('total : %d, correct : %d, incorrect : %d\n' %(len(pred), np.sum(pred == trainY), np.sum(pred != trainY)))

################### ROC CURVE #########################
fpr, tpr, _ = roc_curve(testY, y_score,pos_label = 2)
roc_auc = auc(fpr, tpr)
print roc_auc
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()
