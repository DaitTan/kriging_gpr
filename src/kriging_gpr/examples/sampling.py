
import numpy as np
from numpy.core.fromnumeric import shape
# from testFunction import test_function
# from calculate_robustness import calculate_robustness
from skopt.sampler import Lhs
from skopt.space import Space


def lhs_sampling(number_of_samples: int, region_support: np.array, test_function_dimension: int, rng) -> np.array:
    """Latin Hypercube Sampling: Sample *number_of_samples* points within the *region_support* which has a dimension as mentioned below.

    Args:
        number_of_samples (int): Number of points to sample within the region bounds.
        region_support (np.array): The bounds of the region within which the sampling is to be done.
                                    Region Bounds is M x N x O where;
                                        M = number of regions;
                                        N = test_function_dimension (Dimensionality of the test function);
                                        O = Lower and Upper bound. Should be of length 2;
        test_function_dimension (int): The dimensionality of the region. (Dimensionality of the test function)

    Returns:
        np.array: 3d array with samples between the bounds.
                    Size of the array will be M x N x O
                        M = number of regions
                        N = number_of_samples
                        O = test_function_dimension (Dimensionality of the test function)
    """
    
    assert region_support.shape[1] == test_function_dimension, 'sub_r matrix must be 3-dimensional'
    assert region_support.shape[2] == 2, 'sub_r matrix must be MxNx2'
    assert np.apply_along_axis(lambda bounds: bounds[1] > bounds[0], 2, region_support).all(), 'sub_r Z-pairs must be in increasing order'

    seed_for_lhs = rng.integers(10000000)
    lhs = Lhs(lhs_type="classic", criterion=None)
    space = Space(region_support[0,:,:])
    return np.array([lhs.generate(space.dimensions, number_of_samples, seed_for_lhs)])


def uniform_sampling(number_of_samples: int, region_support: np.array, test_function_dimension: int, rng) -> np.array:
    """Sample *number_of_samples* points within the *region_support* which has a dimension as mentioned below.

    Args:
        number_of_samples (int): Number of points to sample within the region bounds.
        region_support (np.array): The bounds of the region within which the sampling is to be done.
                                    Region Bounds is M x N x O where;
                                        M = number of regions;
                                        N = test_function_dimension (Dimensionality of the test function);
                                        O = Lower and Upper bound. Should be of length 2;
        test_function_dimension (int): The dimensionality of the region. (Dimensionality of the test function)

    Returns:
        np.array: 3d array with samples between the bounds.
                    Size of the array will be M x N x O
                        M = number of regions
                        N = number_of_samples
                        O = test_function_dimension (Dimensionality of the test function)
    """
    
    assert region_support.shape[1] == test_function_dimension, 'sub_r matrix must be 3-dimensional'
    assert region_support.shape[2] == 2, 'sub_r matrix must be MxNx2'
    assert np.apply_along_axis(lambda bounds: bounds[1] > bounds[0], 2, region_support).all(), 'sub_r Z-pairs must be in increasing order'

    raw_samples = np.apply_along_axis(
            lambda bounds: 
                rng.uniform(bounds[0], bounds[1], number_of_samples), 
            2, region_support)

    samples = []
    for sample in raw_samples:
        samples.append(np.transpose(sample))

    return np.array(samples)



# region_support = np.array([[[-1, 1], [-1, 1]]])
# test_function_dimension = 2
# number_of_samples = 10
# seed = 1000
# rng = np.random.default_rng(seed)
# samples_in = uniformSampling(number_of_samples, region_support, test_function_dimension, rng)
# print(samples_in)
# new = samples_in[:,0:3,:]
# new_1 = np.delete(new, (1), axis = 1)
# print(new)
# print(new_1)
# print(samples_in.shape)

# print(new.shape)
# print(new_1.shape)
# # callCount = 0
# R =5
# M = 2
# samples_in_l = samples_in.tolist()[0]
# reshaped_grid = [[samples_in_l[i:i + M]] for i in range(0, (len(samples_in_l)), M)]
# print(np.array(reshaped_grid))

# from testFunction import callCounter

# def test_function(X):  ##CHANGE
#     # return (X[0]**2 + X[1] - 11)**2 + (X[1]**2 + X[0] - 7)**2 - 40 # Himmelblau's
#     # return (100 * (X[1] - X[0] **2)**2 + ((1 - X[0])**2)) - 20 # Rosenbrock
#     return (1 + (X[0] + X[1] + 1) ** 2 * (
#                 19 - 14 * X[0] + 3 * X[0] ** 2 - 14 * X[1] + 6 * X[0] * X[1] + 3 * X[1] ** 2)) * (
#                        30 + (2 * X[0] - 3 * X[1]) ** 2 * (
#                            18 - 32 * X[0] + 12 * X[0] ** 2 + 48 * X[1] - 36 * X[0] * X[1] + 27 * X[1] ** 2)) - 50

# ca = callCounter(test_function)


# samples_out = calculate_robustness(samples_in, ca)

# print(samples_in.shape)
# b = np.array([[[]]])
# print(b.shape)
# a = np.append(samples_in,samples_in, axis=1)

# print(a.shape)
# print(ca.callCount)


# # ca = callCounter(test_function)


# samples_out = calculate_robustness(samples_in, ca)

# print(samples_in.shape)
# b = np.array([[[]]])
# print(b.shape)
# a = np.append(samples_in,samples_in, axis=1)

# print(a.shape)
# print(ca.callCount)
