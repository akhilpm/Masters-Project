'''
2DPCA for feature extraction of MNIST digits dataset 
Author : Akhil P M
'''


from settings import *
from sklearn.ensemble import RandomForestClassifier
import utils


def compute_covariance_matrix(A):
	""" compute the 2D covariance matrix in image space"""

	no_of_images = len(A)
	cov = np.zeros((A.shape[2], A.shape[2]))

	for i in xrange(no_of_images):
		cov = cov + np.dot(np.transpose(A[i]), A[i])

	cov = cov / no_of_images
	return cov	


def extract_feature(A, x):
	""" compute y[i] = A[i]*x for all images """
	no_of_images = len(A)
	features = np.zeros((no_of_images, A.shape[1]))

	for i in xrange(no_of_images):
		features[i] = np.ravel(np.dot(A[i], x))

	return features


def main():
	""" the main function"""

	#set the timer
	start = time.time()

	#load the data
	trainX = np.load('trainX.npy')
	testX = np.load('testX.npy')
	trainY = np.load('trainY.npy')
	testY = np.load('testY.npy')
	print('\n!!! Data Loading Completed !!!\n')

	#generate 2D data
	data_train = utils.generate_2D(trainX)
	data_test = utils.generate_2D(testX)
	ncol = data_train.shape[2]
	features_train = np.zeros((len(data_train), data_train.shape[1]))
	features_test = np.zeros((len(data_test), data_test.shape[1]))

	#get the mean image
	mean_image = utils.get_mean_image(data_train)

	#substract the mean image from all images & center them
	normalized_data = utils.substract_mean(data_train, mean_image)
	data_train = utils.substract_mean(data_train, mean_image)
	data_test = utils.substract_mean(data_test, mean_image)

	#compute the covariance matrix in 2D space
	SA = compute_covariance_matrix(normalized_data)

	#find eigen values & eigen vectors of covariance matrix
	U, s, _ = np.linalg.svd(SA)

	#extract features using 2DPCA
	selected = []
	clf = RandomForestClassifier(n_estimators=300, n_jobs=-1)
	max_acc = 0.0

	for i in xrange(ncol):
		proj_dir = U[:, i].reshape(ncol, 1)
		tempTrainX = extract_feature(data_train, proj_dir)
		tempTestX = extract_feature(data_test, proj_dir)

		clf.fit(tempTrainX, trainY)
		pred = clf.predict(tempTestX)
		acc = accuracy_score(testY, pred)
		print('PC vector %d gives accuracy : %f\n' %(i+1, acc))

		#if acc >=0.1:
		#	selected.append(i)
		#	features_train = features_train + s[i] * tempTrainX
		#	features_test = features_test + s[i] * tempTestX

		if acc > max_acc:
			max_acc = acc
			features_train = np.copy(tempTrainX)
			features_test = np.copy(tempTestX)

	print features_train.shape

	np.save('trainX_feat', features_train)
	np.save('testX_feat', features_test)

	clf.fit(features_train, trainY)
	pred = clf.predict(features_test)
	print('accuracy : %f\n' %accuracy_score(testY, pred))
	#print selected

	print('Test Time : %f Minutes\n' %((time.time()-start)/60))


if __name__ == '__main__':
		main()
