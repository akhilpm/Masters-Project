'''
All modules and global variables can be set in this file
Author: Akhil P M
'''

import numpy as np
import time
from sklearn.datasets.mldata import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from numpy import genfromtxt
from sklearn import svm, datasets
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import ExtraTreesClassifier
