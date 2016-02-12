from settings import *

def display_image(X):
	""" displays the image from the image matrix X """

	im = X.reshape(28, 28)
	temp = plt.imshow(im)
	plt.show()


def generate_2D(X):
	"""	generate 2D image matrix from the 1D vector """

	no_of_images = len(X)
	data = np.zeros((no_of_images, 28, 28))

	for i in xrange(no_of_images):
		data[i] = np.copy(X[i].reshape(28, 28))

	return data


def get_mean_image(data):
	""" returns the mean image """

	no_of_images = len(data)
	mean_im = np.zeros((28, 28))
	for i in xrange(no_of_images):
		mean_im = mean_im + data[i, 0:28, 0:28]

	mean_im = mean_im / no_of_images
	return mean_im

def substract_mean(data, mean_im):
	""" substract the mean image from all images """

	no_of_images = len(data)
	for i in xrange(no_of_images):
		data[i] = data[i] - mean_im

	return data