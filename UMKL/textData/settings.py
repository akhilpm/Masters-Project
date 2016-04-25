'''
All modules and global variables can be set in this file
Author: Akhil P M
'''

import sys
import numpy as np
from time import time
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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectPercentile, f_classif
from datetime import datetime
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import polynomial_kernel

import logging
from optparse import OptionParser

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics.pairwise import euclidean_distances
import cvxopt
from cvxopt import solvers

