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
	KNeighborsClassifier(25),
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
	trainX = np.load('trainX.npy')
	testX = np.load('testX.npy')
	trainY = np.load('trainY.npy')
	testY = np.load('testY.npy')
	print('\n!!! Data Loading Completed !!!\n')

	#iterate over the classifiers
	for name, clf in zip(names, classifiers):
		clf.fit(trainX, trainY)
		pred = clf.predict(testX)
		print('classifier : %s, Accuracy :%f%% ' %(name, accuracy_score(testY, pred)*100))
		print('total : %d, correct : %d, incorrect : %d\n' %(len(pred), np.sum(pred == testY), np.sum(pred != testY)))

	print('Test Time : %f Minutes\n' %((time.time()-start)/60))


if __name__ == '__main__':
	main()	


