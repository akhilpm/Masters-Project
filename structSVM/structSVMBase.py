'''

Basic StructSVM(N-slack formulation) with standard cutting plane algorithm.
Author : Akhil P M

Problem Instance : Multiclass Classification
Kernel type : Linear
Reference : Cutting-Plane training of Structural SVMs, Joachims 2009

'''

import numpy as np
import json
import time
import cvxopt
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from cvxopt import solvers
from sklearn.metrics import accuracy_score

class NSlackSSVM:


	def __init__(self, n_samples, max_iter=1000, verbose=1, eps=1e-3):
		""" initialize the model parameters """

		self.sparam = dict()
		with open('config.json') as config_file:
			self.sparam = json.load(config_file)

		self.C = self.sparam['C']
		self.sizePsi = self.sparam['sizePsi']
		self.num_classes = self.sparam['num_classes']
		#self.w = np.random.rand(sparam['sizePsi'],1)
		self.w = np.zeros((self.sparam['sizePsi'],1))
		self.tempw = np.zeros((self.sparam['sizePsi'],1))
		#self.tempw = np.random.rand(self.sparam['sizePsi'], 1)
		#self.tempw = np.random.rand(self.sparam['sizePsi'])
		#self.tempw[0:self.sizePsi/2] = np.zeros(self.sizePsi/2)
		#self.tempw = self.tempw.reshape(self.sizePsi, 1)
		#print np.sum(self.tempw)
		self.w_changed = False
		self.n = n_samples
		self.max_iter = max_iter
		self.verbose = verbose
		self.eps = eps
		self.alphas = []
		self.losses = []

		

	def solve_n_slack_qp(self, n_samples):
		C = self.C

		#joint_features = [c[3] for sample in self.constraints for c in sample]
		losses = np.array(self.losses)
		no_of_alphas = len(losses)
		tempPsi = np.vstack([item[1] for item in self.w_components])
		#print tempPsi.shape

		P = cvxopt.matrix(np.dot(tempPsi, tempPsi.T))
		q = np.hstack(const[2] for const_iter in self.constraints for const in const_iter)
		q = cvxopt.matrix(q)
		#print q.size
		#print P.size

		G = cvxopt.matrix(np.diag(np.ones(no_of_alphas) * -1))
		h = cvxopt.matrix(np.zeros(no_of_alphas))

		# equality constraint: sum of all alphas must be = C
		A = cvxopt.matrix(np.ones((1, no_of_alphas)))
		b = cvxopt.matrix([C])

		# solve QP model
		cvxopt.solvers.options['feastol'] = 1e-5
		solution = cvxopt.solvers.qp(P, q, G, h, A, b)
		#if solution['status'] != "optimal":
		#	raise ValueError("QP solver failed. Try regularizing your QP.")

		# Lagrange multipliers
		self.alphas = np.ravel(solution['x'])



	def fit(self, X, Y, constraints=None, warm_start=False):
		"""Learn the parameters alpha using cuttting plane method for training"""

		if self.verbose:
			cvxopt.solvers.options['show_progress'] = False

		n_samples = len(X)	

		#part of w is computed for easiness(useful for oneSlack formulation)
		#for i in xrange(self.n):
		#	tempPsi = psi(X[i], Y[i], self.sizePsi)
		#	self.w += np.transpose(tempPsi)
		#self.w = self.w/n

		#initialize all slack variables to zero first
		slacks = np.zeros(n_samples)
		w_components = [] #the alphas and deltaPsi's for all violated constraints are stored here

		if constraints is None:
			self.last_active = [[] for i in range(n_samples)]
			self.objective_curve = []
			self.primal_objective_curve = []

		else:
			objective = self.solve_n_slack_qp(n_samples)

		if not warm_start:
			self.constraints = [[] for i in range(n_samples)]
			self.w_components = []

		for iteration in xrange(self.max_iter):
			if self.verbose > 0:
				print("iteration : %d" %(iteration+1))

			self.w_changed = False

			#find most violated constraint
			for i in xrange(self.n):
				ybar, slack, max_loss, deltaPsi = self.find_most_violated_constraint_margin(X[i], Y[i])

				#print ybar, Y[i], slack, max_loss, np.dot(deltaPsi, self.tempw)
				#chumma = raw_input('wait for key press ')
				#check whether the constraint violation is more than the tolerance level
				#if yes add constraint to the working set
				if (max_loss-np.dot(deltaPsi, self.tempw)) > (slacks[i]+self.eps):
					self.constraints[i].append([ybar, slack, max_loss, deltaPsi])
					self.w_changed = True
					slacks[i] = slack

					#print ybar, Y[i], slack, max_loss, np.dot(deltaPsi, self.tempw), 'from if cond'

					#solve the QP for new alphas
					self.w_components.append([i, deltaPsi])
					self.losses.append(max_loss)					
					self.solve_n_slack_qp(n_samples)


					#calculate tempw
					self.tempw = np.zeros((self.sizePsi, 1))
					tempPsi = np.vstack([item[1] for item in self.w_components])
					tempPsi = np.transpose(tempPsi)
					tempAlphas = np.array(self.alphas)
					
					#print deltaPsi
					#print tempAlphas
					#print tempPsi.shape
					self.tempw = np.sum(tempAlphas*tempPsi, axis=1)
					self.tempw = self.tempw.reshape(self.sizePsi,1)
					#print self.tempw.T

			
			#if no constraints are added stop the optimization process
			if self.w_changed == False:
				break

		print('No. of iterations taken :%d\n' %(iteration+1))		

	def classification_score(self, x, y):
		"""Return an example, label pair's discriminant score."""	
		pass

	def classify_example(self, x):
		"""Returns the classification of an example 'x'."""
		scores = np.zeros(self.num_classes) 

		for c in xrange(self.num_classes):
			Psi = self.psi(x, c)
			scores[c] = np.dot(Psi, self.tempw)

		# Return the label with the max discriminant value.
		return np.argmax(scores)

	def find_most_violated_constraint_margin(self, x, y):
		"""Return ybar associated with x's most violated constraint.

		The find most violated constraint function for margin rescaling.
		The default behavior is that this returns the value from the
		general find_most_violated_constraint function."""

		scores = np.zeros(self.num_classes)
		#dotProd = np.zeros(self.num_classes)

		Psi = self.psi(x, y)
		for i in xrange(self.num_classes):
			Psibar = self.psi(x, i)
			deltaPsi = Psi-Psibar
			#print Psi.shape
			#print self.tempw.shape
			scores[i] = loss(y, i) - np.dot(deltaPsi, self.tempw)
			#dotProd[i] = np.dot(deltaPsi, self.tempw)
			#print scores[i]

		#print scores	
		ybar = np.argmax(scores)
		Psi = self.psi(x, y)
		Psibar = self.psi(x, ybar)
		deltaPsi = Psi-Psibar
		max_loss = loss(y, ybar)
		#print max_loss, y
		#temp = raw_input('enter value')
		slack = max(max_loss - np.dot(deltaPsi, self.tempw), 0)

		return ybar, slack, max_loss, deltaPsi



	def psi(self, x, y):
		"""Returns the combined feature vector Psi(x,y)."""
		Psi = np.zeros(self.sizePsi)
		size = self.sparam['num_features']
		Psi[y*size:(y+1)*size] = x #take x as a column vector
		Psi.reshape((self.sizePsi,1))
		return Psi

	def predict(self, X):
		""" Returns the predicted output of the test dataset"""

		n_samples = X.shape[0]
		predicted = np.zeros(n_samples)

		for i in xrange(n_samples):
			predicted[i] = self.classify_example(X[i])

		return predicted	


def loss(y, ybar, type=2):
	""" Computes the loss associated with predicting ybar as output instead of y """

	#0/1 loss is used by default
	if type==1:
		return (y!=ybar)*1.0
	else:
	    return np.abs(y-ybar)*10.0


def read_dataset():
	digits = datasets.load_digits()
	X = digits.data
	Y = digits.target
	trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.3)
	return trainX, testX, trainY, testY



def main():

	#set the timer
	start = time.time()

	#Load the Dataset
	trainX, testX, trainY, testY = read_dataset()

	#initialize the model
	SSVM = NSlackSSVM(n_samples= trainX.shape[0])

	#fit the model to the data
	SSVM.fit(trainX, trainY)	

	#get the predicted values for the test set
	pred = SSVM.predict(testX)
	#print SSVM.alphas
	print('No. of times margin constraint got violated : %d\n' %len(SSVM.w_components)) 
	print('Classification Accuracy : %f\n' %accuracy_score(testY, pred))
	print('total : %d, correct : %d, incorrect : %d\n' %(len(pred), np.sum(pred == testY), np.sum(pred != testY)))

	print('Execution Time : %f Minutes\n' %((time.time()-start)/60))

	pred = SSVM.predict(trainX)
	print('Accuracy on training set : %f\n' %accuracy_score(trainY, pred))


if __name__ == '__main__':
	main()