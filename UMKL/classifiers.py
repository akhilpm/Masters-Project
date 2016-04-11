import numpy as np
import scipy.io
from scipy.io import loadmat
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import warnings
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression



names = ["Nearest Neighbors", "Linear SVM", "Gaussian SVM", "Polynomial SVM", "Decision Tree",
		 "Random Forest", "AdaBoost Classifier", "Logistic Regression", "Naive Bayes"]


classifiers = [
	KNeighborsClassifier(n_neighbors=15, weights='distance'),
	SVC(kernel="linear", C=3.4),
	SVC(kernel="rbf", C=3.4, gamma=0.1),
	SVC(kernel="poly", C=3.4, degree=2, gamma=0.1),
	DecisionTreeClassifier(),
	RandomForestClassifier(n_estimators=300, n_jobs=-1),
	AdaBoostClassifier(n_estimators=70),
	LogisticRegression(random_state=1, C=0.4),
	GaussianNB()]



def main():
	#ignore all warnings
	#warnings.filterwarnings("ignore")
	start = time.time()

	#load the data
	trainX = np.load('trainX_feat.npy')
	testX = np.load('testX_feat.npy')
	trainY = np.load('trainY_feat.npy')
	testY = np.load('testY_feat.npy')
	print('\n!!! Data Loading Completed !!!\n')


	for name, clf in zip(names, classifiers):
		clf.fit(trainX, trainY)
		pred = clf.predict(testX)
		print('classifier : %s, Accuracy :%f%% ' %(name, accuracy_score(testY, pred)*100))
		print('total : %d, correct : %d, incorrect : %d\n' %(len(pred), np.sum(pred == testY), np.sum(pred != testY)))

		#print(confusion_matrix(testY, pred)) #uncomment to get confusion matrix
	print('Test Time : %f Minutes\n' %((time.time()-start)/60))


if __name__ == '__main__':
	main()
