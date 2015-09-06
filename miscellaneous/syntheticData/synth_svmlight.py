import random
import numpy as np

m=100
w = np.array([0,1,-1])
datafile = open('/home/iistlab/Documents/miniproject/egMulticlass/data/data.dat', 'w+')
tempfile = open('/home/iistlab/Documents/miniproject/egMulticlass/data/temp.dat', 'w+')
representers = m/2
#print data

members = 0
while members < representers:
	#data[i] = np.array([1, random.uniform(0, 10), random.uniform(0, 10)])
	data = [1, random.uniform(0, 10), random.uniform(0, 10)]
	label = np.dot(w, data)
	if label > 0:
		label = 1
		line = str(label) + ' 1:' + str(data[0]) + ' 2:' + str(data[1]) + ' 3:' + str(data[2]) + '\n'
		datafile.writelines(line)
		tempfile.writelines(str(data[1]) + ' ' + str(data[2]) + '\n')
		members +=1	

members = 0
while members < representers:
	#data[i] = np.array([1, random.uniform(0, 10), random.uniform(0, 10)])
	data = [1, random.uniform(0, 10), random.uniform(0, 10)]
	label = np.dot(w, data)
	if label < 0:
		label = 2
		line = str(label) + ' 1:' + str(data[0]) + ' 2:' + str(data[1]) + ' 3:' + str(data[2]) + '\n'
		datafile.writelines(line)
		tempfile.writelines(str(data[1]) + ' ' + str(data[2]) + '\n')
		members +=1	



tempfile.close()
datafile.close()