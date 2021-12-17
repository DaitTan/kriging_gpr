# from ..utils.OK_Rmodel_kd_nugget import OK_Rmodel_kd_nugget
from ..utils.OK_regr import OK_regr
from ..utils.OK_corr import OK_corr

import numpy as np

def OK_Rpredict(gp_model, Xtest, regr_model):
    X = gp_model['X']
    min_X = gp_model['min_X']
    max_X = gp_model['max_X']
    num_samples, dim = X.shape
    theta = gp_model['theta']
    beta = gp_model['beta']
    Z = gp_model['Z']
    L = gp_model['L']
    Rinv = gp_model['Rinv']
    sigma_z = gp_model['sigma_z']
    corr_model = gp_model['corr']
    F = np.ones((num_samples,1))
    Ytrain = gp_model['Y']


    num_samples_test = Xtest.shape[0]

    regr_pred = OK_regr(Xtest,regr_model)

    normal_x_test = (Xtest - min_X)/((max_X-min_X)+1e-6)

    distXpred = np.zeros((dim,num_samples, num_samples_test))

    for h in range(dim):
        for i in range(num_samples):
            for j in range(num_samples_test):
                # distXpred[h,i,j] = normal_x_test[j,h] - X[i,h]
                distXpred[h,i,j] = X[i,h]- normal_x_test[j,h]

    R_pred = OK_corr(2, theta, distXpred)
    # print(R_pred)
    mse = np.full((num_samples_test,1),None)

    f = regr_pred*beta + R_pred.transpose()@(Rinv@(Ytrain-np.ones((num_samples,1))*beta))
    # print(f)
    FRFinv = 1/(F.transpose()@Rinv@F)
    
    Rinv_Rpred = Rinv@R_pred
    # print(FRFinv)
    for r in range(num_samples_test):
        OneMinusFcrossR = 1-(F.transpose()@Rinv_Rpred[:,r])
        
        mse_temp = sigma_z * (1 - R_pred[:,r].transpose()@Rinv_Rpred[:,r] + (OneMinusFcrossR).transpose()@FRFinv@(OneMinusFcrossR))
        mse[r,0] = np.sqrt(mse_temp[0,0])



    # for r in range(num_samples_test):
    #     # sigma_z * (1 - R_pred(:,r)'*Rinv*R_pred(:,r) + (1-F'*Rinv*R_pred(:,r))'*inv(F'*Rinv*F)*(1-F'*Rinv*R_pred(:,r)));
    #     term_1 = (1 - R_pred[:,r].transpose() @ Rinv @ R_pred[:,r])
    #     term_2 = (1-F.transpose() @ Rinv@R_pred[:,r]).transpose()
    #     term_3 = np.linalg.inv(F.transpose() @ Rinv @ F)
    #     term_4 = (1-F.transpose()@Rinv@R_pred[:,r])

    #     mse[r,0] = (sigma_z * (term_1 + term_2@term_3@term_4))[0,0]

    return f, mse