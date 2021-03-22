'''
Created on 20-Feb-2021

@author: Aniket Kumar Tripathi
'''
import numpy as np


# A layer is a tuple that defines number of neurons in each layer
# (input layer, h1,h2, ... hk, output layer)
# The weight matrix for each layer has format (next,curr) next = number of neurons of next layer
# curr = number of neurons of current layer
def create(layers):
    
    NN_weights = []
    NN_biases = []
    a = -1
    b = 2
    # Create a weight matrix for each layer
    for i in range(1, len(layers)):
        # w_ji = weight from input neuron i to ouput neuron j
        weights = a + (b - a) * np.random.random((layers[i], layers[i - 1])).astype(dtype='float64')
        biases = a + (b - a) * np.random.random(layers[i]).astype(dtype='float64')
        NN_weights.append(weights)
        NN_biases.append(biases)
    
    return NN_weights, NN_biases


# Trains the NN and returns the error over each epoch
# The input data set should be in the format (N,d)
# (N,d) N = Number of data points, d = number of dimensions = number of input layer neurons
# Similarly outdata should be in the format (N,d) d = dimensions of output data = number of output layer neurons
def train(NN_weights, NN_biases, layers, datain, dataout, epoch, learning_rate):
    
    error = np.zeros(epoch, dtype='float64')
    N = len(datain)
    l = len(layers)
    
    for i in range(epoch):
        
        # Pass the data set point to NN one by one
        for j in range(N):
            outtrain = fire(NN_weights, NN_biases, datain[j])
            err = (outtrain[l - 1] - dataout[j])
            # Backpropagate error
            backpropagate(NN_weights, NN_biases, outtrain, err, learning_rate)
        # Now test the error
        s = 0
        for j in range(N):
            outtest = fire(NN_weights, NN_biases, datain[j])[l - 1]
            e = (outtest - dataout[j])
            s += (e.T @ e)
        error[i] = s / N
    return error

# l = Number of layers


def backpropagate(NN_weights, NN_biases, out, err, learning_rate):
     
    l = len(NN_weights)
    for i in range(l - 1, -1, -1):
        # print(' out = ', out)
        delta = err * sigmoid_derivative(out[i + 1])
        # print('delta = ', delta)
        err = delta @ NN_weights[i]
        # print('err = ', err)
        delta_weights = np.outer(delta, out[i])
        # print('delta weights = ', delta_weights)
        # print('Unupdated NN weights and biases = ', NN_weights[i], NN_biases[i])  
        NN_weights[i] -= learning_rate * delta_weights
        NN_biases[i] -= learning_rate * delta
        # print('Updated NN weights and biases = ', NN_weights[i], NN_biases[i])
        # print()
    

# Passes a single input point over the NN
# out is the output of all layers. Input for first layer is also included
def fire(NN_weights, NN_biases, inp):
    out = []
    out.append(inp)
    l = len(NN_weights)
    for i in range(l):
        inp = sigmoid((NN_weights[i] @ inp) + NN_biases[i])
        out.append(inp)
        
    return out


# s(x)  = 1/(1 + exp(-x))
def sigmoid(x):
    return np.divide(1, 1 + np.exp(-x))


# provided x is already calculated using sigmoid()
def sigmoid_derivative(x):
    return x * (1 - x)

