import numpy as np

def normalize_data(data):
    """normalizing data

    Args:
        data ([type]): 2D array with shape [num of samples X dimension]

    Returns:
        [normalized]: data noralized between 0 and 1 
    """
    min_data = np.min(data, 0)
    max_data = np.max(data, 0)

    normalized_data = (data - min_data)/((max_data-min_data)+1e-6)
    return normalized_data, min_data, max_data