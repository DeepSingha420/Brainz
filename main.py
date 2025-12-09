import numpy as np
from neuron import NN

if __name__ == "__main__":
    
    neural = NN()

    print("Random starting weights:")
    print(neural.weights)

    #test = np.array([1, 0, 0])

    t_in = np.array([[0,0,1],
                     [1,0,1],
                     [1,1,1],
                     [0,1,1]])
    
    t_out = np.array([[0,1,1,0]]).T

    print("Training...")
    neural.train(t_in,t_out,10000)

    print("New weights after training:")
    print(neural.weights)

    new_test = np.array([1,0,0])

    print(f"Thinking with input {new_test}:")
    result = neural.think(new_test)
    print(result)

