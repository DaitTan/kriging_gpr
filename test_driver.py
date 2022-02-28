from example_files.testFunction import callCounter
from example_files.sampling import uniform_sampling, lhs_sampling
from example_files.calculate_robustness import calculate_robustness
import sys

from src.kriging_model.interface.OK_Rmodel_kd_nugget import OK_Rmodel_kd_nugget
from src.kriging_model.interface.OK_Rpredict import OK_Rpredict
from src.kriging_model.utils.OK_regr import OK_regr
from src.kriging_model.utils.OK_corr import OK_corr

import numpy as np


def test_function(X):  ##CHANGE
    return (X[0]**2 + X[1] - 11)**2 + (X[1]**2 + X[0] - 7)**2 - 40 # Himmelblau's


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


# The parameter_a takes care of the error of semi-definite matrices.
# Change the value to get rid of the error
# To get to root of details, look into
# Line 76 in src.krigin_model.interface.OK_Rmodel_kd_nugget.py
gp_model = OK_Rmodel_kd_nugget(Xtrain, data_out, 0, 2, parameter_a=10)



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

# z = test_output.reshape((xGrid.shape[0], xGrid.shape[1])).transpose()

z = test_output.reshape((xGrid.shape[0], xGrid.shape[1])).transpose()
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=1, cols=1, specs = [[{'type': 'surface'}]])

print(f.shape)
print(pred_ci_lower.shape)


mean_output = f.reshape((xGrid.shape[0], xGrid.shape[1])).transpose()
pred_ci_lower = pred_ci_lower.reshape((xGrid.shape[0], xGrid.shape[1])).transpose()
pred_ci_upper = pred_ci_upper.reshape((xGrid.shape[0], xGrid.shape[1])).transpose()

fig.add_trace(
    go.Surface(x=xGrid, y=yGrid, z=mean_output, colorscale='ylorrd', showscale=False, name = 'Predicted Output'),
    row=1, col=1)
fig.add_trace(
    go.Surface(x=xGrid, y=yGrid, z=pred_ci_lower, colorscale='earth', showscale=False, name = 'Lower CB'),
    row=1, col=1)
fig.add_trace(
go.Surface(x=xGrid, y=yGrid, z=pred_ci_upper, colorscale='greens', showscale=False, name = 'Upper CB'),
row=1, col=1)
fig.update_layout(showlegend = True, legend_title_text='Plots')
fig.show()

