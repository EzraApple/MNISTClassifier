import numpy as np


def l2_normalize_samples(data):
    """
    Normalizes each row by that rows L2 norm
    :param data: data
    :return: normalized data
    """
    data = data/np.linalg.norm(data, axis=1)
    return data
