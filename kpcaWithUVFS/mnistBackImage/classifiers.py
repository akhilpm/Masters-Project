'''

A set of classifiers to run on the data
Author : Akhil P M

'''

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from settings import *

names = ["Nearest Neighbors", "Linear SVM", "Decision Tree", "Random Forest",
		"AdaBoost Classifier","Logistic Regression", "Naive Bayes"]


classifiers = [
	KNeighborsClassifier(n_neighbors=25, weights='distance'),
	SVC(kernel="linear", C=3.4),
	DecisionTreeClassifier(),
	RandomForestClassifier(n_estimators=300, n_jobs=-1),
	AdaBoostClassifier(n_estimators=70),
	LogisticRegression(random_state=1, C=0.4),
	GaussianNB()]

def main():

	#set the timer
	start = time.time()

	#load the data
	trainX = np.load('trainX_feat.npy')
	testX = np.load('testX_feat.npy')
	trainY = np.load('trainY_feat.npy')
	testY = np.load('testY_feat.npy')
	print('\n!!! Data Loading Completed !!!\n')

	#iterate over the classifiers
	for name, clf in zip(names, classifiers):
		clf.fit(trainX, trainY)
		pred = clf.predict(testX)
		print('classifier : %s, Accuracy :%f%% ' %(name, accuracy_score(testY, pred)*100))
		print('total : %d, correct : %d, incorrect : %d\n' %(len(pred), np.sum(pred == testY), np.sum(pred != testY)))
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

	clf = RandomForestClassifier(n_estimators=1400, n_jobs=-1)
	clf.fit(trainX, trainY)
	pred = clf.predict(testX)
	print accuracy_score(testY, pred)
	print('total : %d, correct : %d, incorrect : %d\n' %(len(pred), np.sum(pred == testY), np.sum(pred != testY)))
	'''
	print('Test Time : %f Minutes\n' %((time.time()-start)/60))


if __name__ == '__main__':
	main()	


