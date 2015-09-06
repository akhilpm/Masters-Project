from sklearn.datasets.mldata import fetch_mldata
import numpy as np
from sklearn.cross_validation import train_test_split

#Load the Digit Data Set
mnist=fetch_mldata('MNIST original')
trainX,testX,trainY,testY=train_test_split(mnist.data,mnist.target,test_size=0.3)
mtrain,ntrain=trainX.shape
mtest,ntest=testX.shape

#Append one to the first column of the training data
ones=np.ones((mtrain,1), dtype=int)
trainX=np.append(ones,trainX,axis=1)
ones=np.ones((mtest,1), dtype=int)
testX=np.append(ones,testX,axis=1)
ntrain=ntest=ntrain+1

# make digits range as 1-10
trainY=trainY+1 
testY=testY+1
num_classes=len(np.unique(trainY))
theta=np.random.random_sample((ntrain,num_classes))


def costFunction(trainX,theta,trainY):
	m,n=trainX.shape
	#for logistic regression	
	hyp=np.dot(trainX,theta)
	hyp=1/(1+np.exp(-hyp))
	cost=trainY*np.log(hyp)+(1-trainY)*np.log(1-hyp)
	jval=-np.sum(cost)/m
	grad=np.sum((hyp-trainY)*trainX,axis=0)
	grad=np.transpose(grad)
	grad=grad/m
	return jval,grad

	#take all theta(k) values as a vector to compute numerator in P(yi=k/xi;theta)
	#by a vectorized implementation
	theta_dash=np.transpose(theta) # to make theta(k*n)
	theta_vec=theta_dash[trainY-1]
	temp=np.sum(theta_vec*trainX,axis=1)
	temp=np.exp(-temp)
	hyp_sum=np.zeros((m,1))
	for i in np.arange(m):
		hyp_sum[i]=np.sum(np.dot(trainX[i],theta))
	#calculate the hypothesis value
	hyp=np.log(temp/hyp_sum)
	jval=-np.sum(hyp)
	num_classes=len(np.unique(trainY))
	#compute the gradient wrt to all theta_k
	grad=np.random.random_sample((num_classes,n))
	for k in np.arange(num_classes):
		gradK=1*(trainY==k+1)
		theta_vec=theta_dash[k]
		temp=np.sum(theta_vec*trainX,axis=1)
		temp=np.exp(-temp)
		hyp=temp/hyp_sum
		new_row=np.sum(trainX*(gradK-hyp),axis=0)
		grad[k]=np.copy(new_row)
	#convert grad into a vector	
	grad=np.hstack(grad)	
	return jval,grad			