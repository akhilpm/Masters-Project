# test train split

import random
import numpy as np

no_of_classes = 4

lines = open('/home/iistlab/Documents/miniproject/egMulticlass/data/vehicle.dat').readlines()
train_size = int(len(lines)*.6)
no_of_reps = int(train_size/no_of_classes)
#print train_size
#print no_of_reps
random.shuffle(lines)

representers = np.zeros(no_of_classes) 
testfile = open('/home/iistlab/Documents/miniproject/egMulticlass/data/test.dat', 'w+')
trainfile = open('/home/iistlab/Documents/miniproject/egMulticlass/data/train.dat', 'w+')


for line in lines:
	label = int(line[0])
	if representers[label-1] < no_of_reps:
		trainfile.writelines(line)
		representers[label-1] += 1
	else:
		testfile.writelines(line)	

#close all files
testfile.close()
trainfile.close()
