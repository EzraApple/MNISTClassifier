import numpy as np


def l2_normalize_samples(data):
    """
    Normalizes each row by that rows L2 norm
    :param data: data
    :return: normalized data
    """
    data = data/np.linalg.norm(data, axis=1)[:, None]
    return data


def normalize_features(data):
    """
    subtracts mean from each feature in data (columns are features),
    then divides by standard deviation. Each feature will now have mean 0,
    and standard deviation 1
    :param data: sample data
    :return: normalized sample data
    """
    n, d = data.shape
    feature_means = data.mean(axis=0)
    feature_std = data.std(axis=0)
    data = data - feature_means
    data = data / feature_std
    return data
