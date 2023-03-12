import numpy as np


class SoftMaxRegression:

    def fit(self, data, labels, iterations):
        num_classes = np.unique(labels).shape[0]
        n, d = data.shape
        weights = np.zeros((data.shape[1], num_classes))


    def predict(self, data):
        pass

    def evaluate(self, data):
        pass