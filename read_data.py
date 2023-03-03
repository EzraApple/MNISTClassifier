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
    return data


def read_raw_mnist(data):
    """
    Function specifically to split xxxx1 x 785 default matrix
    - removes .csv headers in row 1
    -splits first column off to have xxxxx x 1 labels
    - rest is xxxxx x 784 data points
    :param data: xxxx1 x 785 data
    :return: data, labels
    """
    remove_headers = data[1:, :]
    data_points = remove_headers[:, 1:]
    labels = remove_headers[:, 0:1]
    return data_points, np.squeeze(labels)
