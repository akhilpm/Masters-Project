import numpy as np
import numpy.linalg as linalg
import numpy.matlib

gama = 1 #kernel width parameter for gaussian kernel

alpha = np.random.uniform(0, 1, 51)
while True:
	temp = np.random.uniform(-1, 0, 49)
	if abs(linalg.norm(alpha) - linalg.norm(temp)) < 0.01:
		break

alpha = np.append(alpha, temp)
alpha = alpha.reshape((100, 1))
#print alpha

""" an array of support points created randomly """
funX = np.random.uniform(low = -1, high = 1, size = (100, 2))

""" build the dataset """
rawfile = open('/home/iistlab/Documents/miniproject/egMulticlass/synthetic/raw.dat', 'w+')
score = np.zeros((1000,1))
label = 1

for i in xrange(1000):
	data_point = np.random.uniform(-1, 1, 2)#np.random.random((1, 2))
	data_new  = numpy.matlib.repmat(data_point, 100, 1)
	kvalues = np.exp(-gama * np.power(linalg.norm(funX - data_new, axis=1), 2))
	score[i] = np.sum(alpha * kvalues)
	line = str(label) + ' 1:' + str(data_point[0]) + ' 2:' + str(data_point[1]) + '\n'
	rawfile.writelines(line)

rawfile.close()
score = score - np.average(score)

rawfile = open('/home/iistlab/Documents/miniproject/egMulticlass/synthetic/raw.dat', 'r')
datafile = open('/home/iistlab/Documents/miniproject/egMulticlass/synthetic/synthetic.dat', 'w+')
i=0
for line in rawfile:
	if np.sign(score[i]) > 0:
		label = 1
	else:
		label = 2
	line[0] = str(label)
	datafile.writelines(line)
	i += 1


rawfile.close()
datafile.close()

alphafile = open('/home/iistlab/Documents/miniproject/egMulticlass/synthetic/alpha.dat', 'w+')
for a in alpha:
	alphafile.writelines(a)
alphafile.close()


svfile = open('/home/iistlab/Documents/miniproject/egMulticlass/synthetic/svectors.dat', 'w+')
for a in funX:
	svfile.writelines(a)
svfile.close()