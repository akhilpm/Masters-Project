'''
All modules and global variables can be set in this file
Author: Akhil P M
'''

import sys
import numpy as np
import time
import warnings
import gc
from sklearn.datasets.mldata import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from numpy import genfromtxt
from sklearn import svm, datasets
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import NearestNeighbors
import cvxopt
from cvxopt import solvers

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectPercentile, f_classif
from datetime import datetime
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import euclidean_distances

from joblib import Parallel, delayed  
import multiprocessing

#a simple utility for logging all print statements
class Tee(object):
	def __init__(self, *files):
		self.files = files
		#print files
	def write(self, obj):
		for f in self.files:
			f.write(obj)

f = open(sys.argv[2], 'w')
backup = sys.stdout
sys.stdout = Tee(sys.stdout, f)