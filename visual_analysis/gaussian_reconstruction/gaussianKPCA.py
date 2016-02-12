import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.cm as cm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectPercentile, f_classif


def main():

	#set the timer
	start = time.time()

	#load the data
	trainX = np.load('trainX.npy')
	testX = np.load('testX.npy')
	trainY = np.load('trainY.npy')
	testY = np.load('testY.npy')
	print('\n!!! Data Loading Completed !!!\n')

	#get the 1st digit zero and plot it
	zero = trainX[14].reshape(28, 28)
	plt.imshow(zero, cmap=cm.Greys_r)
	plt.savefig("original"+str(trainY[14])+".png")
	#plt.show()

	#apply kpca
	kpca = KernelPCA(kernel='rbf', gamma=1, fit_inverse_transform=True)
	kpca.fit(trainX[0:3000])
	trainX_kpca = kpca.transform(trainX)
	testX_kpca = kpca.transform(testX)

	#do inverse transform and plot the result
	orig = kpca.inverse_transform(trainX_kpca)
	img = orig[14].reshape(28, 28)
	plt.imshow(img, cmap=cm.Greys_r)
	plt.savefig("reconstructed"+str(trainY[14])+".png")
	#plt.show()

	selector = SelectPercentile(f_classif, percentile=5)
	selector.fit(trainX_kpca, trainY)
	trainX = selector.transform(trainX_kpca)
	testX = selector.transform(testX_kpca)

	#fit a classifier
	parameters = {'n_neighbors' : list(np.arange(15)+1)}
	clf = GridSearchCV(KNeighborsClassifier(weights='distance', n_jobs=-1), parameters)
	clf.fit(trainX, trainY)

	pred = clf.predict(testX)
	print accuracy_score(testY, pred)
	print confusion_matrix(testY, pred)
	#print(clf.best_params_)
	print('total : %d, correct : %d, incorrect : %d\n' %(len(pred), np.sum(pred == testY), np.sum(pred != testY)))

	print('Test Time : %f Minutes\n' %((time.time()-start)/60))


if __name__ == '__main__':
	main()
