import numpy as np


def l2_normalize_samples(data):
    """
    Normalizes each row by that rows L2 norm
    :param data: data
    :return: normalized data
    """
    data = data/np.linalg.norm(data, axis=1)[:, None]
    return data


def standardize_features(data):
    """
    returns a copy of data with standardized
    features (columns)
    :param data: sample data
    :return: normalized sample data
    """
    copy = np.copy(data)
    feature_means = copy.mean(axis=0)
    feature_std = copy.std(axis=0)
    copy -= feature_means
    copy /= feature_std
    copy[np.isnan(copy)] = 0
    return copy
