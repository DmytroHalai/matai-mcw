import numpy as np
from utils import sigmoid

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            w = np.random.uniform(-1, 1, (layer_sizes[i + 1], layer_sizes[i]))
            b = np.random.uniform(-1, 1, (layer_sizes[i + 1], 1))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, inputs):
        a = inputs
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            a = sigmoid(np.dot(w, a) + b)
        return np.dot(self.weights[-1], a) + self.biases[-1]
