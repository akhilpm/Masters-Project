'''

feature selection STRATEGY is implemented here
inputs : X, y, feature ranking vector, indices of the features accroding to the ranking

Author : Akhil P M

'''




def select_features(X, importances, indices):

	#STRATEGY : select features having importance greater than some threshold
	need = (importances > 0.00035)
	new_indx = indices[need]
	#print new_indx
	newX = X[:, new_indx]

	return newX


def norm_values(norms, importances, indices):
	''' extracts the norm values of selected features powered by their n value'''

	need = (importances > 0.0005)
	new_indx = indices[need]

	return norms[new_indx]
