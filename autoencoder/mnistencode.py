'''
Sparse Autoencoder 
Author: Akhil P M
Courtesy: UFLDL stanford, Siddharth Agarwal
mail: akhilpm135@gmail.com
'''

import numpy as np
import time
import math
import scipy
import scipy.io
import matplotlib.pyplot as plt
import scipy.optimize
import sys
from sklearn.datasets.mldata import fetch_mldata
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import preprocessing
import matplotlib.gridspec as gridspec

no_of_func_calls = 1
verbosity_level = 0


class SparseAutoencoder(object):

	def __init__(self, input_size, hidden_size, lambdaa, rho, beta):

		""" initialize the parameters of the Autoencoder"""

		self.input_size = input_size #no of input units
		self.hidden_size = hidden_size #no of hidden units
		self.lambdaa = lambdaa #network weight regularization factor
		self.rho = rho # desired average activation of hidden units
		self.beta = beta # weight of sparsity penalty term

		#limits used to unroll theta into weights and biases
		self.limit0 = 0
		self.limit1 = hidden_size * input_size
		self.limit2 = 2 * hidden_size * input_size
		self.limit3 = 2 * hidden_size * input_size + hidden_size
		self.limit4 = 2 * hidden_size * input_size + hidden_size + input_size

		#initialize biase and weights
		rand = np.random.RandomState(23455)
		r = math.sqrt(6)/math.sqrt(input_size + hidden_size + 1)

		W1 = np.asarray(rand.uniform(low=-r, high=r, size=(hidden_size, input_size)))
		W2 = np.asarray(rand.uniform(low=-r, high=r, size=(input_size, hidden_size)))
		b1 = np.zeros((hidden_size,1))
		b2 = np.zeros((input_size,1))

		#unroll all parameters into a single vector for optimization
		self.theta = np.concatenate((W1.flatten(), W2.flatten(),
					b1.flatten(), b2.flatten()))

		print('======Autoencoder initialized===========')

	def sparse_autoencoder_cost(self,theta,trainX):
		'''computes the cost in an iteration'''

		global verbosity_level
		#m = no of attributes, n= no of datapoints
		m,n = trainX.shape
		total_cost=0.0


		"""extract weights and biases from theta"""
		W1 = theta[self.limit0 : self.limit1].reshape(self.hidden_size, self.input_size)
		W2 = theta[self.limit1 : self.limit2].reshape(self.input_size, self.hidden_size)
		b1 = theta[self.limit2 : self.limit3].reshape(self.hidden_size,1)
		b2 = theta[self.limit3 : self.limit4].reshape(self.input_size,1)

		"""perform a forward pass"""
		act_hidden_layer = sigmoid(np.dot(W1, trainX) + b1)
		act_output_layer = sigmoid(np.dot(W2, act_hidden_layer) + b2)
		if verbosity_level == 2:
			print('activations of hidden and output layers computed')

		"""estimate avg activation of hidden units"""
		rho_avg = np.sum(act_hidden_layer, axis=1)/n

		diff = act_output_layer-trainX
		sum_of_squares_error = 0.5 * np.sum(np.square(diff))/n
		weight_deacay = 0.5 * self.lambdaa * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
		KL_divergence = self.beta * np.sum(self.rho * np.log(self.rho/rho_avg) + 
						(1-self.rho) * np.log((1-self.rho)/(1-rho_avg)))

		total_cost = sum_of_squares_error + weight_deacay + KL_divergence

		"""compute error in hidden layer and output layer"""		
		delta3 = np.multiply(diff, np.multiply(act_output_layer, 1-act_output_layer))
		KL_div_grad = self.beta*(-(self.rho/rho_avg) + ((1-self.rho)/(1-rho_avg)))
		delta2 = np.multiply(np.dot(np.transpose(W2),delta3) + 
			np.transpose(np.matrix(KL_div_grad)), np.multiply(act_hidden_layer,1-act_hidden_layer))


		"""compute the gradient"""
		W1_grad = np.dot(delta2, np.transpose(trainX))
		W2_grad = np.dot(delta3, np.transpose(act_hidden_layer))
		b1_grad = np.sum(delta2, axis=1)
		b2_grad = np.sum(delta3, axis=1)

		W1_grad = W1_grad/n + self.lambdaa*W1
		W2_grad = W2_grad/n + self.lambdaa*W2
		b1_grad = b1_grad/n
		b2_grad = b2_grad/n

		W1_grad = np.array(W1_grad)
		W2_grad = np.array(W2_grad)
		b1_grad = np.array(b1_grad)
		b2_grad = np.array(b2_grad)


		"""unroll the gradients into a single vector"""
		theta_grad = np.concatenate((W1_grad.flatten(), W2_grad.flatten(),
			b1_grad.flatten(), b2_grad.flatten()))

		if verbosity_level >= 1:
			print('grad J(theta) computed')

		return [total_cost,theta_grad]



def sigmoid(x):
	return (1/(1 + np.exp(-x)))	

def normalizeDataset(dataset):

	""" Remove mean of dataset """
	dataset = dataset - np.mean(dataset)
    
	""" Truncate to +/-3 standard deviations and scale to -1 to 1 """
	
	std_dev = 3 * np.std(dataset)
	dataset = np.maximum(np.minimum(dataset, std_dev), -std_dev) / std_dev
    
	""" Rescale from [-1, 1] to [0.1, 0.9] """
	dataset = (dataset + 1) * 0.4 + 0.1
	return dataset


def load_dataset(num_patches, patch_size):
	'''utility function to load data set'''

	global verbosity_level
	print('======loading dataset=======\n')


	mnist = fetch_mldata('MNIST original')
	sss = StratifiedShuffleSplit(mnist.target, 1, test_size=0.1, train_size=20000, random_state=0)
	for train_index, test_index in sss:
		trainX, testX = mnist.data[train_index], mnist.data[test_index]
		trainY, testY = mnist.target[train_index], mnist.target[test_index]

	no_of_images  = trainX.shape[0]	
	""" the dataset is originally read as dictionary, convert it to an array.
		the resulting array is of shape[512,512,10]. 
		no of images=10
		image size = 512*512(gray scale)
	"""
	#dataset is of shape [64*10,000]
	dataset = np.zeros((patch_size*patch_size, num_patches))

	"""Randomly sample images"""
	rand = np.random.RandomState(23455)
	image_number = rand.randint(no_of_images, size = num_patches)

	for i in xrange(num_patches):
		""""get the patch indices """
		index3 = image_number[i]

		""""extract patch from original image"""
		dataset[:,i] = trainX[index3]

	if verbosity_level==2:
		print('=========patches extracted========\n')
	"""normalize the dataset(min max feature scaling is used)"""
	#transpose 'dataset' to form attributes as columns of the matrix, since scaling
	#is to be done featurewise
	if verbosity_level==2:
		print('***********scaling features to [0.1, 0.9] range***********\n')

	#dataset = normalizeDataset(dataset)	
	dataset = dataset / 255.0
	#dataset = np.transpose(dataset) # newsize = 10,000*64
	#min_max_scaler = preprocessing.MinMaxScaler()
	#dataset = min_max_scaler.fit_transform(dataset)
	#dataset = np.transpose(dataset) #transpose to 64*10,000

	print('======loading dataset : completed ========\n')
	return dataset



def visualizeW1(opt_W1, input_patch_size, hidden_patch_size):

	""" Add the weights as a matrix of images """

	figure, axes = plt.subplots(nrows = hidden_patch_size, ncols = hidden_patch_size)
	#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)		
	index = 0

	for axis in axes.flat:

		""" Add row of weights as an image to the plot """   
		image = axis.imshow(opt_W1[index, :].reshape(input_patch_size, input_patch_size),
			cmap = plt.cm.gray, interpolation = 'nearest')
		axis.set_frame_on(False)
		axis.set_axis_off()
		index += 1

	""" Show the obtained plot """
	plt.show()


def execute_sparse_autoencoder(level):
	'''main function'''

	"""set values for the parameters of Autoencoder"""
	start_time = time.time()
	input_patch_size  = 28 #size of sampled image patches
	hidden_patch_size = 14 #size of representative image patches
	rho = .01 			  # sparsity parameter(desired avg activation of hidden units)
	num_patches = 10000   #no of training patches
	lambdaa = 0.01      #weight decay parameter
	beta = 0.1              # weight of the sparsity penalty term
	max_iterations = 400  #maximum iterations for optimization
	global verbosity_level  #set the verbosity level
	verbosity_level = level
	error = 0.0

	input_size = input_patch_size * input_patch_size
	hidden_size = hidden_patch_size * hidden_patch_size

	"""load the dataset and preprocess it"""
	data_train = load_dataset(num_patches, input_patch_size)

	"""initialize the Autoencoder"""
	encoder = SparseAutoencoder(input_size, hidden_size, lambdaa, rho, beta)

	"""do gradient checking to verify the correctness of implenentation"""
	#error = scipy.optimize.check_grad(func, gradient, encoder.theta, encoder, data_train)
	#print('error in gradient : %f\n' %(error))

	if error < 0.001:
		print('++++++++++++++gradient checking passed+++++++++++')
	else:
		sys.exit('***********gradient checking failed***********')


	"""do the optimization using L-BFGS algoritm"""
	opt_solution = scipy.optimize.minimize(encoder.sparse_autoencoder_cost,
					encoder.theta, args=(data_train,), method='L-BFGS-B', 
                    jac=True, options={'maxiter': max_iterations, 'disp' : True})

	print('optimization success : %r\n' %(opt_solution.success)) 
	opt_theta = opt_solution.x
	opt_W1 = opt_theta[encoder.limit0 : encoder.limit1].reshape(hidden_size,input_size)

	print('execution time : %f' %(time.time() -start_time))

	"""visualize W1"""
	visualizeW1(opt_W1, input_patch_size, hidden_patch_size)


if __name__ == '__main__':
	execute_sparse_autoencoder(0)