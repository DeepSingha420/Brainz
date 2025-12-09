import numpy as np
from activation import sigmoid, sigmoid_d

class NN():
    def __init__(self, layer_size):
        np.random.seed(1)

        self.layer_size = layer_size
        input_size = layer_size[0]
        hidden_size = layer_size[1]
        output_size = layer_size[2]
        
        self.weights1 = np.random.rand(input_size,hidden_size)
        self.weights2 = np.random.rand(hidden_size,output_size)


    def train(self, X, y, epochs = 1500):
        self.y = y
        for epoch in range(epochs):
            self.forward(X)
            self.back()

    def forward(self, X):
        #inputs = inputs.astype(float)
        self.input = X
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        return self.output

    def back(self):
            
            weight2_d = np.dot(self.layer1.T, (2*(self.y - self.output)*sigmoid_d(self.output)))

            weight1_d = np.dot(self.input.T, (np.dot(2*(self.y - self.output)*sigmoid_d(self.output), self.weights2.T)*sigmoid_d(self.layer1)))

            self.weights1 += weight1_d
            self.weights2 += weight2_d

    def save_weights(self, filename):
        np.savez(filename, weights1=self.weights1, weights2=self.weights2)
        print(f"Weights saved to {filename}")
    
    def load_weights(self, filename):
        data = np.load(filename)
        self.weights1 = data['weights1']
        self.weights2 = data['weights2']

        self.input_size = self.weights1.shape[0]
        self.hidden_size = self.weights1.shape[1]
        self.output_size = self.weights2.shape[1]

        print(f"Weights loaded from {filename}")