import numpy as np
import math
from .OK_corr import OK_corr


def OK_Rlh_kd_nugget(params, num_samples, dim, D_X, Y, regr, corr_model, delta):
    # params = np.reshape(params, (dim,1))
    # if np.min(params[0:dim,0]) <= 0.001:
    #     f = math.inf
    #     return math.inf
    
    theta = np.reshape(params, (dim,1))
    # print(theta)
    R = OK_corr(corr_model, theta, D_X)
    R = R + delta * (np.eye(R.shape[0], R.shape[1]))
    CR = R
    U = (np.linalg.cholesky(CR)).transpose()
    
    L  = np.transpose(U)
    Linv = np.linalg.inv(L)
    Sinv = Linv.transpose() @ Linv
    

    beta = np.linalg.inv(np.transpose(regr) @ Sinv @ regr)@(np.transpose(regr) @ (Sinv @ Y))
    
    sigma_z = (1/num_samples) * (np.transpose(Y - (regr @ beta)) @ Sinv @ (Y - (regr@beta)))
    
    f = num_samples * (np.log(sigma_z)) + np.log(np.linalg.det(R))
    eps = 1e-15
    if np.isnan(f[0,0]) or np.isinf(f[0,0]):
        f = num_samples * (np.log(sigma_z+eps)) + np.log(np.linalg.det(R)+eps)
    
    return f[0,0]