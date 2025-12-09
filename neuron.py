import numpy as np
from activation import sigmoid, sigmoid_d

class NN():
    def __init__(self):
        np.random.seed(1)
        self.weights = 2 * np.random.random((3, 1)) - 1
    
    def think(self, inputs):
        inputs = inputs.astype(float)
        output = sigmoid(np.dot(inputs, self.weights))
        return output

    def train(self, t_in, t_out, iterations):
        for iteration in range(iterations):
            output = self.think(t_in)
            error = t_out - output
            adjustment = np.dot(t_in.T, error*sigmoid_d(output))
            self.weights += adjustment