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
    
    neural = NN(layer_size=[3,10,1])

    print("Training...")
    neural.train(X,y,epochs=2000)
    print(f"Before Saving Weights: \n{neural.forward(X)}")

    neural.save_weights("nn_weights.npz")

    print("Weights saved.")

    print("Reseting weights and loading from file...")
    neural2 = NN(layer_size=[3,10,1])

    print(f"Without Training: \n{neural2.forward(X)}")

    print("Loading weights...")
    neural2.load_weights("nn_weights.npz")

    print("Loaded weights.")

    print(f"After Loading Weights: \n{neural2.forward(X)}")



