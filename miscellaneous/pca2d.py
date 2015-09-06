import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

epsilon = np.exp(-5)

# load the data and make the mean zero
x = np.loadtxt('/home/akhi/progs/pcaData.txt')
plt.figure(1)
plt.scatter(x[0,:], x[1,:])

avg = np.mean(x, 0)
x = x - np.matlib.repmat(avg, 2, 1)


"""compute covariance matrix, and then eigen values and eigen vectors of covariance
matrix using singular value decomposition"""
sigma = np.dot(x, np.transpose(x)) / x.shape[1] 
U, S, V = np.linalg.svd(sigma) #np.dot(np.dot(U,np.diag(S)),V) gives the sigma back

#plot eigne vectors and verirfy their direction
plt.plot([0, U[0,0]],[0, U[1,0]])
plt.plot([0, U[0,1]],[0, U[1,1]])
plt.show()

"""compute the transformation to the new basis U
we can choose to retain k components in this stage"""
xRot = np.dot(np.transpose(U), x) # U'*x;
xTilde = np.dot(np.transpose(U[:,0:1]), x) # retains only one component


#PCA whitening
plt.figure(2)
xPCAWhite =  np.dot(np.diag(1 / np.sqrt(S + epsilon)), xRot)
# scale xPCAWhite[1](2nd row) with 1e+34 to get the same plot as in matlab,
# since the range of values is very small. if you didn't scale python will 
# apptoximate it to xero and you will get a straight line parallel to x - axis 
xPCAWhite[1] = xPCAWhite[1] * np.exp(34)
plt.scatter(xPCAWhite[0,:], xPCAWhite[1,:])
plt.show()


#ZCA whitening
plt.figure(3)
xZCAWhite = np.dot(U, xPCAWhite)
plt.scatter(xZCAWhite[0,:], xZCAWhite[1,:])
plt.show()
