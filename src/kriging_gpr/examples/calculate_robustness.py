import numpy as np
# from testFunction import test_function

def calculate_robustness(samples_in: np.array, test_function)->np.array:
    """Compute the fitness (robustness) of the given sample.

    Args:
        samples_in (np.array): Samples points for which the fitness is to be computed.

    Returns:
        np.array: Fitness (robustness) of the given sample
    """
    
    if len(samples_in.shape) == 1:
        # case when only one sample is to be assessed
        samples_out = np.array([test_function(samples_in)])
    else:
        # case when samples are to be assessed
        samples_out = np.apply_along_axis(lambda sample: test_function(sample),2,samples_in)
        # print(np.apply_along_axis(lambda sample: test_function(sample, callCount),2,samples_in))
    return samples_out