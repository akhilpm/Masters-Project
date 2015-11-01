import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import decomposition




def my_kernel(x, y):
    """
    We create a custom kernel:

                 (2  0)
    k(x, y) = x  (    ) y.T
                 (0  1)
    """
    M = np.array([[2, 0], [0, 1.0]])
    return np.dot(np.dot(x, M), y.T)

def compute_J(N, theta):

  if N == 0:
    return np.pi - theta
  elif N == 1:
    return np.sin(theta) + (np.pi - theta) * np.cos(theta)
  elif N == 2:
    return 3*np.sin(theta)*np.cos(theta) + (np.pi - theta)*(1 + 2*pow(np.cos(theta), 2))
  elif N == 3:
    return 4*pow(np.sin(theta), 3) + 15*np.sin(theta)*pow(np.cos(theta), 2) + \
    (np.pi- theta)*(9*pow(np.sin(theta),2)*np.cos(theta) + 15*pow(np.cos(theta),3))
  else:    
    return np.zeros(theta.shape)


def arc_cosine(X, Y):
  """param = a vector of n(degree) values at each layer """
  print X.shape
  param = np.array([3,2,2,1])
  no_of_layers = len(param)


  M = np.dot(X, Y.T)
  temp1 = np.diag(np.dot(X, X.T))
  temp2 = np.diag(np.dot(Y, Y.T)) 

  for i in xrange(no_of_layers):
    norm_matrix = np.outer(temp1,temp2) #the matix of k_xx, and K_yy's
    theta = np.arccos( np.maximum( np.minimum(M/np.sqrt(norm_matrix), 1.0), -1.0))
    n_l = param[i]
    M = np.multiply(np.power(norm_matrix, n_l/2.0), compute_J(n_l, theta)) / np.pi


    if i < no_of_layers-1:
      zero1 = np.zeros(len(temp1))
      zero2 = np.zeros(len(temp2))
      temp1 = np.multiply(np.power(temp1, n_l), compute_J(n_l, zero1)) / np.pi
      temp2 = np.multiply(np.power(temp2, n_l), compute_J(n_l, zero2)) / np.pi


  return M    


# import some data to play with
iris = datasets.load_iris()
X = iris.data
Y = iris.target


pca = decomposition.PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)

h = 0.2  # step size in the mesh

# we create an instance of SVM and fit out data.
#clf = svm.SVC(kernel=my_kernel)
trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.3)
testX = testX.astype(dtype = 'float32')
clf = svm.SVC(kernel=arc_cosine, C=0.05)
clf.fit(trainX, trainY)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
plt.title('IRIS dataset classification using SVM with Arc-cosine kernel')
plt.axis('tight')
#plt.show()

pred = clf.predict(testX)
print accuracy_score(testY, pred)
print('total : %d, correct : %d, incorrect : %d\n' %(len(pred), np.sum(pred == testY), np.sum(pred != testY)))

plt.show()