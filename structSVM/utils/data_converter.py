'''

Converts datapoints from both dense to sparse(for SVMlight) and sparse to 
dense(for sklearn based LIBSVM and feature engg) representations.

Author : Akhil P M

'''

import numpy as np
from sklearn import datasets


def elewise_normal_to_svmlight(X, Y, svmLightFile):
	#you should open in append mode
	datafile = open(svmLightFile, 'w+')
	
	line = str(int(Y))
	for i in xrange(len(X)):
		if X[i]!=0:
				line = line + ' ' + str(i+1) +':'+ str(X[i])

	line = line + '\n'
	datafile.writelines(line)

	datafile.close()	

def normal_to_svmlight(X, Y, svmLightFile):

	try:
		no_of_attributes = X.shape[1]
	except :
		elewise_normal_to_svmlight(X, Y, svmLightFile)
		return

	#you should open in append mode
	datafile = open(svmLightFile, 'w+')
	for i in xrange(len(X)):
		line = str(int(Y[i]))
		for j in xrange(no_of_attributes):
			if X[i][j]!=0:
				line = line + ' ' + str(j+1) +':'+ str(X[i][j])

		line = line + '\n'
		datafile.writelines(line)

	datafile.close()	



def svmlight_to_normal(X, Y, normalFile):


def main():

	iris = datasets.load_iris()
	#X = iris.data
	#Y = iris.target
	#X = np.array([[0,2.4,3.55],[2.1,0,5.966],[5.4,6.3654,0]])
	#Y = np.ones(3)
	X = np.array([0,2.4,3.55])
	Y = 1

	normal_to_svmlight(X, Y, 'data.dat')


if __name__ == '__main__':
	main()
