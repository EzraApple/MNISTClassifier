import numpy as np
import matplotlib.pyplot as plt


class LDAClassifier:
    """
    LDA is a subset of gaussian discriminant analysis
    where an assumption is made that all class normal
    distributions share a covariance matrix, which is
     a 'pooled within-class covariance matrix' aka
     get each classes cov matrix and average them
    """

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

        for label in range(self.num_classes):

            # isolate class data
            class_indices = np.where(labels.reshape(-1,) == label)
            class_data = np.squeeze(data[class_indices, :])

            # compute priors
            self.priors[label] = class_data.shape[0] / n

            # compute mean for this digit
            self.means[label, :] = class_data.mean(axis=0)

            # scale/add this digit's cov matrix
            self.cov += np.cov(class_data.T)
        self.cov = self.cov/10

        # to avoid singular covariance matrix,
        # add min singular value to diagonals --> PSD
        u, sig, v = np.linalg.svd(self.cov)
        self.cov = self.cov + (sig.min()) * np.eye(*self.cov.shape)

    def predict(self, data):
        """
        Give the predicted label for data,
        argmax of ldf return
        :param data: data - n sample points
        :return: predicted data point - n x 1
        """
        probabilities = self.ldf(data)
        predictions = np.argmax(probabilities, axis=1)
        return predictions

    def ldf(self, data):
        """
        Evaluates the linear discriminant function
        at the datapoint(s) and returns (data.size) x (num_classes)
        matrix of estimated probs.

        https://people.eecs.berkeley.edu/~jrs/189/lec/09.pdf
        page 49

        :param data: data points
        :return: ldf for each class on data points
        """
        shape = data.shape[0], self.num_classes
        results = np.zeros(shape)

        for label in range(self.num_classes):
            mean = self.means[label, :]
            prior = self.priors[label]

            # ldf of form w * x + alpha
            w = np.linalg.solve(self.cov.T, mean)
            alpha = -0.5 * w.T.dot(mean) + np.log(prior)

            ldf = (data.dot(w) + alpha)
            results[:, label] = ldf

        return results

    def error(self, data, labels):
        """
        Computes and compares predicted values to
        actual labels and returns the error as
        count(predicted == actual)/count(samples)
        :return: error
        """
        predicted = self.predict(data)
        error = 1 - (predicted == labels).mean()
        return error

    def visualize_cov(self):
        plt.imshow(self.cov)
        plt.title(f"Covariance Matrix Visualization")
        plt.colorbar()
        plt.show()


class QDAClassifier:

    def fit(self, data, labels):
        """
        Get class means, covariances, and priors
        :param data: training set data
        :param labels: training set labels
        :return: None
        """
        n, d = data.shape
        self.classes = np.unique(labels)
        self.num_classes = self.classes.shape[0]

        self.priors = np.zeros((self.num_classes, 1))
        self.means = np.zeros((self.num_classes, d))
        self.covs = np.zeros((self.num_classes, d, d))

        for label in range(self.num_classes):
            # isolate class data
            class_indices = np.where(labels.reshape(-1, ) == label)
            class_data = np.squeeze(data[class_indices, :])

            # compute priors
            self.priors[label] = class_data.shape[0] / n

            # compute mean for this digit
            self.means[label, :] = class_data.mean(axis=0)

            # add this digit's cov matrix
            class_cov = np.cov(class_data.T)
            self.covs[label] = class_cov

    def predict(self, data):
        """
        Give the predicted label for data,
        argmax of qdf return
        :param data: data - n sample points
        :return: predicted data point - n x 1
        """
        probabilities = self.qdf(data)
        predictions = np.argmax(probabilities, axis=1)
        return predictions

    def qdf(self, data):
        """
        Evaluates the linear discriminant function
        at the datapoint(s) and returns (data.size) x (num_classes)
        matrix of estimated probs.

        https://people.eecs.berkeley.edu/~jrs/189/lec/09.pdf
        page 47.

        :param data: data points
        :return: qdf for each class on data points
        """
        shape = data.shape[0], self.num_classes
        results = np.zeros(shape)

        for label in range(self.num_classes):
            mean = self.means[label]
            prior = self.priors[label]
            cov = self.covs[label]

            centered_data = data - mean
            u, sig, v = np.linalg.svd(cov)
            cov = cov + (sig.min() * 100) * np.eye(*cov.shape)

            w = np.linalg.solve(cov, centered_data.T)
            alpha = -0.5 * np.linalg.det(cov) + prior
            qdf = -0.5 * (centered_data * w.T).sum(-1) + alpha
            results[:, label] = qdf

        return results

    def error(self, data, labels):
        """
        Computes and compares predicted values to
        actual labels and returns the error as
        count(predicted == actual)/count(samples)
        :return: error
        """
        predicted = self.predict(data)
        error = 1 - (predicted == labels).mean()
        return error

    def visualize_cov(self, label):
        """
        visualize covariance matrix for label
        :param label: which label to visualize
        :return: None
        """
        plt.imshow(self.covs[label])
        plt.title(f"Covariance Matrix Visualization: Label {label}")
        plt.colorbar()
        plt.show()

