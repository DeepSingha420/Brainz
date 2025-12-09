import numpy as np
from activation import sigmoid, sigmoid_d

class NN():
    def __init__(self,x,y):
        np.random.seed(1)

        self.input = x
        self.y = y
        self.output = np.zeros(self.y.shape)

        self.weights1 = np.random.rand(self.input.shape[1],4)
        self.weights2 = np.random.rand(4,1)


    def forward(self):
        #inputs = inputs.astype(float)
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def back(self):
            
            weight2_d = np.dot(self.layer1.T, (2*(self.y - self.output)*sigmoid_d(self.output)))

            weight1_d = np.dot(self.input.T, (np.dot(2*(self.y - self.output)*sigmoid_d(self.output), self.weights2.T)*sigmoid_d(self.layer1)))

            self.weights1 += weight1_d
            self.weights2 += weight2_d