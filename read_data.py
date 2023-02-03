import numpy as np


def separate_labels(dataset):
    """
    Function to separate the labels from the data

    :param dataset: data set with first item in each row as label,
    the rest as data entries
    :return: two numpy arrays: labels, and data
    """
    labels = dataset[:, 0]
    data = dataset[:, 1:]
    return labels, data


def mnist_training_set():
    """
    Fetches the training set from the .csv file
    :return: numpy array of mnist dataset
    """
    training_set = np.genfromtxt("data_csv/mnist_train.csv", delimiter=",")
    return training_set[1:]


def mnist_test_set():
    """
        Fetches the test set from the .csv file
        :return: numpy array of mnist dataset
        """
    test_set = np.genfromtxt("data_csv/mnist_test.csv", delimiter=",")
    return test_set[1:]
