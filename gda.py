import numpy as np


class LDAClassifier:
    """
    LDA is a subset of gaussian discriminant analysis
    where an assumption is made that all class normal
    distributions share a covariance matrix, which is
     a 'pooled within-class covariance matrix' aka
     get each classes cov matrix and average them
    """
    def __init__(self):
        self.labels = None
        self.data = None
        self.cov = None
        self.means = None
        self.priors = None
        self.classes = None
        self.num_classes = None

    def fit(self, data, labels):
        """
        Get parameters for distributions:
        -mean for each class
        -shared covariance matrix
        -prior probabilities
        :param data: training data
        :param labels: training labels
        :return: None
        """
        self.data = data
        self.labels = labels
        n, d = data.shape
        self.classes = np.unique(labels)
        self.num_classes = self.classes.shape[0]
        self.priors = np.zeros((self.num_classes, 1))
        self.means = np.zeros((self.num_classes, d))
        self.cov = np.zeros((d, d))

        for label in self.classes:

            # isolate class data
            class_data = data[np.where(labels == label), :]
            print(class_data.shape)

            # compute priors
            self.priors[label] = class_data.shape[0] / n

            # compute mean for this digit
            self.means[label, :] = class_data.mean(axis=0)

            # scale/add this digit's cov matrix
            self.cov += np.cov(class_data.T)/10

    def predict(self, data):
        """
        Give the predicted label for data
        :param data: data - n sample points
        :return: predicted data point - n x 1
        """
        pass

    def ldf(self, data):
        """
        Evaluates the linear discriminant function
        at the datapoint(s) and returns (data.size) x (num_classes)
        matrix of estimated probs.
        :param data: data points
        :return: ldf for each class on those data points
        """
        pass

    def error(self, test_d, test_l):
        """
        Computes and compares predicted values to
        actual labels and returns the error as
        count(predicted == actual)/count(samples)
        :return: error of test set
        """
        pass






