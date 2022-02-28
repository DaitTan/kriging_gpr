import numpy as np

def OK_regr(data, regr_model):
    # Call the regression function for the MNEK model
    # X - design locations for the simulation inputs, size [k, d], k points with d
    # dimensions 
    # regr_model - the underlying regression model for the mean function:
    # regr_model = 0: constant mean function;
    # regr_model = 1: linear mean function;
    # regr_model = 2: quadratic mean function;

    num_samples, dim = data.shape
    
    if regr_model == 0:
        regr = np.ones((num_samples,1))
    elif regr_model == 1:
        regr = np.hstack((np.ones((num_samples,1)),data))
    elif regr_model == 2:
        mid = ((dim+1) * (dim+2))/2.0
        regr = np.hstack((np.ones((num_samples,1)), data, np.zeros((num_samples, (mid-dim-1)))))
        j = dim + 1
        q = dim
        for i in range(dim):
            regr[:,j+np.arange(0,q,1)] = np.multiply(np.repmat(X[:,i],1,q),X[:,i:n])
    return regr