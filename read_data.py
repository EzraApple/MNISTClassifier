import numpy as np


def csv_to_array(path):
    """
    Reads .csv file at 'path' into array
    :param path: path to file
    :return: ndarray
    """
    array = np.genfromtxt(path, delimiter=',')
    return array


def csv_to_npz(path, file):
    """
    Reads a .csv file and saves it as a compressed .npz
    :param path: path to .csv file
    :param file: name of new file or open file destination
    :return: None
    """
    array = csv_to_array(path)
    np.savez_compressed(file, array)


def npz_to_data(path):
    """
    Reads .npz file at 'path' into data
    :param path: path to file
    :return:data
    """
    data = np.load(path)

csv_to_npz("data_csv/mnist_train.csv", "mnist_train")
csv_to_npz("data_csv/mnist_test.csv", "mnist_test")