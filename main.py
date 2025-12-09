import numpy as np
from neuron import NN

if __name__ == "__main__":
    
   

    #print("Random starting weights:")
    #print(neural.weights)

    #test = np.array([1, 0, 0])

    X = np.array([[0,0,1],
                     [0,1,1],
                     [1,0,1],
                     [1,1,1]])
    
    y = np.array([[0],[1],[1],[0]])
    
    neural = NN(X,y)

    print("Training...")
    for i in range(1500):
        neural.forward()
        neural.back()

    print("Output:")
    print(neural.output)


