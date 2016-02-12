import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import manifold, datasets, decomposition, ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score


def visualize_digits(trainX):
	""" plot the raw digits """

	n_img_per_row = 10
	img = np.zeros((30 * n_img_per_row, 30 * n_img_per_row))
	print img.shape
	for i in range(n_img_per_row):
		ix = 30 * i + 1
		for j in range(n_img_per_row):
			iy = 30 * j + 1
			img[ix:ix + 28, iy:iy + 28] = np.transpose(trainX[i * n_img_per_row + j].reshape((28, 28)))

	plt.imshow(img, cmap=plt.cm.binary)
	plt.xticks([])
	plt.yticks([])
	plt.title('MNIST RAND visualization')
	plt.show()


# Scale and visualize the embedding vectors
def plot_embedding(X, Y, title=None):
	x_min, x_max = np.min(X, 0), np.max(X, 0)
	X = (X - x_min) / (x_max - x_min)

	plt.figure()
	ax = plt.subplot(111)
	for i in range(X.shape[0]):
		plt.text(X[i, 0], X[i, 1], str(Y[i]), color=plt.cm.Set1(Y[i] / 10.), fontdict={'weight': 'bold', 'size': 9})

	'''	
	if hasattr(offsetbox, 'AnnotationBbox'):
		# only print thumbnails with matplotlib > 1.0
		shown_images = np.array([[1., 1.]])  # just something big
		for i in range(X.shape[0]):
			dist = np.sum((X[i] - shown_images) ** 2, 1)
			if np.min(dist) < 4e-3:
				# don't show points that are too close
				continue
			shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),X[i])
			ax.add_artist(imagebox)
	'''		
	plt.xticks([]), plt.yticks([])
	if title is not None:
		plt.title(title)


def compute_tSNE_embedding(trainX, testX):
	# t-SNE embedding of the digits dataset
	print("Computing t-SNE embedding")
	tsne = manifold.TSNE(n_components=10, init='pca', random_state=0)

	t0 = time.time()
	X_train = tsne.fit_transform(trainX)
	print('Training set embedding Time : %f Minutes\n' %((time.time()-t0)/60))

	t0 = time.time()
	X_test = tsne.fit_transform(testX[0:10000])
	print('Testing set embedding Time : %f Minutes\n' %((time.time()-t0)/60))

	np.save('X_train', X_train)
	np.save('X_test', X_test)

	return X_train, X_test



def compute_spectral_embedding(trainX, testX):
	''' SpectralEmbedding of digits dataset '''

	print("Computing Spectral embedding")
	embedder = manifold.SpectralEmbedding(n_components=2, random_state=0, eigen_solver="arpack")
	t0 = time.time()

	X_train = embedder.fit_transform(trainX)
	print('Training set embedding Time : %f Minutes\n' %((time.time()-t0)/60))

	t0 = time.time()
	X_test = embedder.fit_transform(testX)
	print('Testing set embedding Time : %f Minutes\n' %((time.time()-t0)/60))

	np.save('X_train', X_train)
	np.save('X_test', X_test)

	return X_train, X_test




def main():

	#set the timer
	start = time.time()

	#load the data
	trainX = np.load('trainX_feat.npy')
	print trainX.shape
	testX = np.load('testX_feat.npy')
	trainY = np.load('trainY_feat.npy')
	testY = np.load('testY_feat.npy')
	testX = testX[0:10000]
	testY = testY[0:10000]


	#plot images of the digits
	#visualize_digits(trainX)


	#X_train = np.load('X_train.npy')
	#print X_train.shape
	#X_test = np.load('X_test.npy')
	X_train, X_test = compute_tSNE_embedding(trainX, testX)

	#plot_embedding(X_train, trainY, "tsne embedding of the digits (time %.2fs)" %(time.time() - start))
	#plt.show()

	parameters = {'n_neighbors' : list(np.arange(20)+1)}
	clf = GridSearchCV(KNeighborsClassifier(weights='distance', n_jobs=-1), parameters)
	#clf = svm.SVC(kernel=kernel.arc_cosine, cache_size=2048)
	clf.fit(X_train, trainY)


	pred = clf.predict(X_test)
	print accuracy_score(testY, pred)
	print confusion_matrix(testY, pred)
	#print(clf.best_params_)
	print('total : %d, correct : %d, incorrect : %d\n' %(len(pred), np.sum(pred == testY), np.sum(pred != testY)))

	print('Test Time : %f Minutes\n' %((time.time()-start)/60))

if __name__ == '__main__':
	main()