import numpy as np


class Neuron:
    def __init__(self, input_size, fn):
        """
        Randomly initializes weights, bias of zero,
         and passes in activation function.
        :param input_size: size of input for this layer
        :param fn: activation function for this neuron
        """
        self.w = np.random.uniform(-1, 1, input_size)
        self.b = 0
        self.fn = fn

    def activate(self, x):
        """
        Calculates fn(x * w + b) where * is dot product.
        :param x: input vector
        :return: scalar activation value
        """
        activation = self.fn(np.dot(self.w, x) + self.b)
        return activation

    def __repr__(self):
        return f"Neuron: {self.w.shape[0]} to 1"


class Layer:
    def __init__(self, input_size, size, fn):
        """
        Creates neurons for this layer and saves
        some instance vars for __repr__.
        :param input_size: size of input vector
        :param size: number of neurons
        :param fn: activation function for neurons
        """
        self.size = size
        self.input_size = input_size
        self.fn = fn
        self.neurons = [Neuron(input_size, fn) for _ in range(size)]

    def pass_through(self, x):
        """
        Creates output ndarray of neuron activations.
        :param x: layer input
        :return: output vector
        """
        output = np.array([n.activate(x) for n in self.neurons])
        return output

    def __repr__(self):
        return f"Layer of {self.size} Neurons with input size {self.input_size}"


class OutputLayer:
    def __init__(self, intput_size, output_size):
        """
        Creates layer of linear neurons and saves output size for repr
        :param intput_size: Size of input vector
        :param output_size: size of softmax output vector
        """
        self.output_size = output_size
        self.neurons = [Neuron(intput_size, lambda x: x) for i in range(output_size)]

    def pass_through(self, x):
        dots = np.array([n.activate(x) for n in self.neurons])
        exp = np.exp(dots)
        probs = exp * 1/sum(exp)
        return probs


class Model:
    def __init__(self):
        self.layers = []

    def add_layer(self, input_size, size, fn):
        """
        Adds layer with parameters to self.layers.
        :param input_size: size of input vector for layer
        :param size: number of neurons in this layer
        :param fn: activation fn for this layer
        """
        self.layers.append(Layer(input_size, size, fn))

    def add_output(self, input_size, output_size):
        """
        Creates a softmax output layer
        :param input_size: size of input vector
        :param output_size: size of output vector
        """
        self.layers.append(OutputLayer(input_size, output_size))

    def predict(self, x):
        for layer in self.layers:
            x = layer.pass_through(x)
        return x
