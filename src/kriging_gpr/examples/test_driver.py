from testFunction import callCounter
from sampling import uniform_sampling, lhs_sampling
from calculate_robustness import calculate_robustness

from OK_Rmodel_kd_nugget import OK_Rmodel_kd_nugget
from OK_Rpredict import OK_Rpredict
from OK_regr import OK_regr
from OK_corr import OK_corr

import numpy as np



def test_function(X):  ##CHANGE
    return (X[0]**2 + X[1] - 11)**2 + (X[1]**2 + X[0] - 7)**2 - 40 # Himmelblau's
    # return (100 * (X[1] - X[0] **2)**2 + ((1 - X[0])**2)) - 20 # Rosenbrock
    # return (1 + (X[0] + X[1] + 1) ** 2 * (
    #             19 - 14 * X[0] + 3 * X[0] ** 2 - 14 * X[1] + 6 * X[0] * X[1] + 3 * X[1] ** 2)) * (
    #                    30 + (2 * X[0] - 3 * X[1]) ** 2 * (
    #                        18 - 32 * X[0] + 12 * X[0] ** 2 + 48 * X[1] - 36 * X[0] * X[1] + 27 * X[1] ** 2)) - 50




# import pprint
# pp = pprint.PrettyPrinter(indent = 5)
# pp.pprint(m)



region_support = np.array([[[-5., 5.], [-5., 5.]]])
test_function_dimension = 2
number_of_samples = 200
seed = 1000
rng = np.random.default_rng(seed)
test_func = callCounter(test_function)

samples_in = lhs_sampling(number_of_samples, region_support, test_function_dimension, rng)
data_out = np.transpose(calculate_robustness(samples_in, test_func))

test_samples = lhs_sampling(20, region_support, test_function_dimension, rng)
test_output = np.transpose(calculate_robustness(test_samples, test_func))


Xtest = test_samples[0]
Xtrain = samples_in[0]
Ytrain = data_out


regr_model = 0
corr_model = 2

# print(Xtrain)
# print(Ytrain)

gp_model = OK_Rmodel_kd_nugget(Xtrain, data_out, 0, 2, 5)

# import pprint
# pp = pprint.PrettyPrinter(indent = 3)
# pp.pprint(gp_model)

# print(f.shape)
# # np.hstack((f[:,0],test_output[:,0]))
# import matplotlib.pyplot as plt
# # from mpl_toolkits import mplot3d
# # ax = plt.axes(projection='3d')
# plt.plot(f, 'r', label="Prediction")
# plt.plot(pred_ci_lower,'--b', label="CI Lower")
# plt.plot(pred_ci_upper,'--y', label="CI Upper")
# plt.plot(test_output,'g', label="True Value")
# plt.show()


x = np.linspace(-5,5,20)
y = np.linspace(-5,5,20)
xGrid, yGrid = np.meshgrid(y, x)

x_ = xGrid.reshape((1, xGrid.shape[0]*xGrid.shape[1],1))
y_ = yGrid.reshape((1, yGrid.shape[0]*yGrid.shape[1],1))

test_input = np.concatenate((x_, y_),2)
test_output =  np.transpose(calculate_robustness(test_input, test_func))


f, mse = OK_Rpredict(gp_model, test_input[0], 0)

pred_ci_lower = f - 1.96*mse
pred_ci_upper = f + 1.96*mse


z = test_output.reshape((xGrid.shape[0], xGrid.shape[1])).transpose()
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=1, cols=1, specs = [[{'type': 'surface'}]])

# z = test_output.reshape((xGrid.shape[0], xGrid.shape[1])).transpose()
# fig.add_trace(
#     go.Surface(x=xGrid, y=yGrid, z=z, colorscale='viridis', showscale=False, name = 'True Output'),
#     row=1, col=1)

# ci_lower = pred_ci_lower.reshape((xGrid.shape[0], xGrid.shape[1])).transpose()
# fig.add_trace(
#     go.Surface(x=xGrid, y=yGrid, z=ci_lower, colorscale='rdbu', showscale=False, name = 'Lower CI bound'),
#     row=1, col=1)

# ci_upper = pred_ci_upper.reshape((xGrid.shape[0], xGrid.shape[1])).transpose()
# fig.add_trace(
#     go.Surface(x=xGrid, y=yGrid, z=ci_upper, colorscale='ylgnbu', showscale=False, name = 'Upper CI bound'),
#     row=1, col=1)


mean_output = f.reshape((xGrid.shape[0], xGrid.shape[1])).transpose()
fig.add_trace(
    go.Surface(x=xGrid, y=yGrid, z=mean_output, colorscale='ylorrd', showscale=False, name = 'Predicted Output'),
    row=1, col=1)

fig.update_layout(showlegend = True, legend_title_text='Plots')
fig.show()

