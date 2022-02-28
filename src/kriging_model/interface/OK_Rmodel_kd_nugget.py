import numpy as np
np.set_printoptions(threshold=np.inf)
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.linalg import eigh

from ..utils.OK_regr import OK_regr
from ..utils.OK_corr import OK_corr
from ..utils.OK_Rlh_kd_nugget import OK_Rlh_kd_nugget
from ..utils.normalize_data import normalize_data

def OK_Rmodel_kd_nugget(data_in, data_out, regr_model, corr_model, parameter_a):
    # print(Xtrain)
    num_samples, dim = data_in.shape
    # print("**************************************************************************************")
    # print(data_in)
    # print(data_out)
    

    normal_data, min_data, max_data = normalize_data(data_in)
    # print(normal_data)

    tmp = 0
    D_x = np.zeros((dim,num_samples,num_samples))
    temp_d_x = np.zeros((num_samples*num_samples, dim))

    for h in range(dim):
        hh = 0
        for i in range(num_samples):
            for l in range(num_samples):
                D_x[h,i,l] = normal_data[i,h] - normal_data[l,h]
                temp_d_x[hh,h] = normal_data[i,h] - normal_data[l,h]
                hh = hh+1


    regr = OK_regr(normal_data,regr_model)
    beta_0 = np.linalg.lstsq(((np.transpose(regr) @ regr)), (np.transpose(regr) @ data_out), rcond=None)
    
    beta_0 = beta_0[0]
    sigma_z0 = np.var(data_out-(regr @ beta_0))
    theta_0 = np.zeros((dim,1))

    # print(Xtrain)
    # print("*******************")
    # print(data_out)
    # print("*******************")
    # print(beta_0)
    # print("*******************")
    # print(sigma_z0)
    # print("*******************")
    # print(theta_0)
    # print("*******************")
    
    # print("****************************************************************************************")
    # print("****************************************************************************************")
    # print(np.mean(np.abs(temp_d_x),0)+1e-10)
    # # print("**************")
    # # print((np.mean(np.abs(temp_d_x),0)**(-1*corr_model)))
    # # print("**************")
    # # print((np.log(2)/dim))
    # # print("**************")
    # print("****************************************************************************************")
    # print("****************************************************************************************")

    if corr_model == 0 or corr_model == 3:
        theta_0[:,0] = 0.5
    else:
        theta_0[:,0] = (np.log(2)/dim) * ((np.mean(np.abs(temp_d_x),0)+1e-10)**(-1*corr_model))

    

    # 
    corr = OK_corr(corr_model, theta_0, D_x)
    

    a = parameter_a
    # eigen_v, eigen_vec = np.linalg.eig(corr)
    cond_ = np.linalg.cond(corr, p=2)
    exp_ = np.exp(a)
    # delta_lb = np.maximum(((np.max(eigen_v) * (cond_ - exp_))/(cond_ * (exp_ - 1))),0)

    if (np.allclose(corr, corr.T, rtol=1e-7, atol=1e-10)):
        eigen_v = eigh(corr, eigvals_only=True, subset_by_index=[corr.shape[0]-1, corr.shape[0]-1])[0]

    
    delta_lb = np.maximum(((eigen_v * (cond_ - exp_))/(cond_ * (exp_ - 1))),0)
    lob_sigma_z = 0.00001*sigma_z0
    lob_theta = 0.001*np.ones((dim,1)) 

    lower_bound_theta = np.ndarray.flatten(lob_theta)
    upper_bound_theta = np.full(lower_bound_theta.shape, np.inf)

    options = {'maxiter' : 1000000}
    lob = [lob_theta]
    bnds =  Bounds(lower_bound_theta, upper_bound_theta)
    fun = lambda p_in: OK_Rlh_kd_nugget(p_in, num_samples, dim, D_x, data_out, regr, corr_model, delta_lb)
    
    # fun([3.07342516, 3.08924981])
    
    params = minimize(fun, np.ndarray.flatten(theta_0), method = 'Nelder-Mead', bounds = bnds, options = options)

    theta = np.reshape(params.x, (dim,1))

    R = OK_corr(corr_model, theta, D_x)

    CR = (R+delta_lb*np.eye(R.shape[0],R.shape[1]))

    U0 = np.linalg.cholesky(CR).transpose()
    CR=U0
    L  = np.transpose(U0)
    D_L = np.transpose(U0)
    Linv = np.linalg.inv(L)
    Rinv = Linv.transpose() @ Linv

    beta = np.linalg.inv(np.transpose(regr) @ Rinv @ regr)@(np.transpose(regr) @ (Rinv @ data_out))
    beta_v = np.linalg.inv(np.transpose(regr) @ Rinv @ regr)@(np.transpose(regr) @ Rinv)
    sigma_z = (1/num_samples) * (np.transpose(data_out - (regr @ beta)) @ Rinv @ (data_out - (regr@beta)))
    # print("*******************")
    # print(beta)
    # print("*******************")
    # print(beta_v)
    # print("*******************")
    # print(sigma_z)
    # print("*******************")

    M_model={'sigma_z' :  sigma_z,
    'min_X' : min_data,
    'max_X' : max_data,
    'regr' :  regr,
    'beta' : beta,
    'beta_v' : beta_v,
    'theta' : theta,
    'X' : normal_data,
    'corr' : corr_model,
    'L' : L,
    'D_L' : D_L,
    'Z' : np.linalg.lstsq(L,(data_out-regr@beta), rcond=None),
    'Z_v' : np.linalg.lstsq(L,(np.eye(np.max(data_out.shape))-regr@beta_v), rcond=None),
    'Z_m' : np.linalg.inv(L),
    'DZ_m' : np.linalg.inv(D_L),
    'Rinv' : Rinv,
    'nugget' : delta_lb,
    'Y' : data_out}
    return M_model



# region_support = np.array([[[-10, 10], [-10, 10]]])
# test_function_dimension = 2
# number_of_samples = 10
# seed = 1000
# rng = np.random.default_rng(seed)
# test_func = callCounter(test_function)

# samples_in = uniform_sampling(number_of_samples, region_support, test_function_dimension, rng)
# data_out = np.transpose(calculate_robustness(samples_in, test_func))

# Xtrain = samples_in[0]
# regr_model = 1
# corr_model = 2
# num_samples, dim = samples_in[0].shape

# m= OK_Rmodel_kd_nugget(Xtrain, data_out, regr_model, corr_model)

# import pprint
# pp = pprint.PrettyPrinter(indent = 5)
# pp.pprint(m)