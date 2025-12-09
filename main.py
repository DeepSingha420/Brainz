import numpy as np
from neuron import NN
from loader import load_data

if __name__ == "__main__":
    
   

    #print("Random starting weights:")
    #print(neural.weights)

    #test = np.array([1, 0, 0])

    print("Reading Data")
    
    X, y = load_data("mnist_test.csv")

    in_shape = X.shape[1]
    tar_shape = y.shape[1]

    print(f"Input Shape: {in_shape}")
    print(f"Target Shape: {tar_shape}")


    neural = NN(layer_size=[in_shape,100,tar_shape])

    print("Training...")
    neural.train(X,y,epochs=5, rate=0.1)

    #neural.save_weights("mnist_weights.npz")
    #print("Weights saved.")

    neural.load_weights("mnist_weights.npz")

    '''test_img = X[1]
    tar = y[0]

    pred = neural.forward(np.array(test_img))



    print(f"Test Result:")
    print(f"Predicted: {np.round(pred,2)}")
    print(f"Target: {tar}")

    print(f"Predicted Number: {np.argmax(pred)}")'''

    print("Accuracy Test:")
    score = []

    for i in range(len(X)):
        tar = np.argmax(y[i])
        output = neural.forward(X[i])
        pred = np.argmax(output)

        if tar == pred:
            score.append(1)
        else:
            score.append(0)
    sc = np.array(score)

    #accuracy = (sum(score)/len(score)) * 100

    accuracy = sc.sum() / sc.size * 100

    print(f"Accuracy: {accuracy} %")




