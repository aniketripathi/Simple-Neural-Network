'''
Created on 17-Feb-2021

@author: Aniket Kumar Tripathi
'''

import NeuralNetwork as NN
import matplotlib.pyplot as plt
import numpy as np


def AND(epoch, learning_rate):
    
    # Define the input data and correct output
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    a = np.array([[0], [0], [0], [1]])
   
    layers = (2, 1)
    NN_weights, NN_biases = NN.create(layers)
    error = NN.train(NN_weights, NN_biases, layers, data, a, epoch, learning_rate)
    return error
    

def OR(epoch,learning_rate):
   
    # Define the input data and correct output
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    a = np.array([[0], [1], [1], [1]])
   
    layers = (2, 1)
    NN_weights, NN_biases = NN.create(layers)
    error = NN.train(NN_weights, NN_biases, layers, data, a, epoch, learning_rate)
    return error
    
    
def NAND(epoch,learning_rate):
    
    # Define the input data and correct output
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    a = np.array([[1], [1], [1], [0]])
   
    layers = (2, 1)
    NN_weights, NN_biases = NN.create(layers)
    error = NN.train(NN_weights, NN_biases, layers, data, a, epoch, learning_rate)
    return error

    
def NOR(epoch,learning_rate):
    
    # Define the input data and correct output
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    a = np.array([[1], [0], [0], [0]])
   
    layers = (2, 1)
    NN_weights, NN_biases = NN.create(layers)
    error = NN.train(NN_weights, NN_biases, layers, data, a, epoch, learning_rate)
    return error


def XOR(epoch,learning_rate):
    # Define the input data and correct output
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    a = np.array([[0], [1], [1], [0]])
    
    layers = (2, 2, 1)
    NN_weights, NN_biases = NN.create(layers)
    error = NN.train(NN_weights, NN_biases, layers, data, a, epoch, learning_rate)
    return error

def XNOR(epoch,learning_rate):
    # Define the input data and correct output
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    a = np.array([[1], [0], [0], [1]])
    
    layers = (2, 2, 1)
    NN_weights, NN_biases = NN.create(layers)
    error = NN.train(NN_weights, NN_biases, layers, data, a, epoch, learning_rate)
    return error



    
def compare():
    
    epoch1 = 500;
    learning_rate = 0.1
     
    errors1 = np.zeros((4,epoch1))
    errors1[0] = AND(epoch1,learning_rate)
    errors1[1] = OR(epoch1,learning_rate)
    errors1[2] = NAND(epoch1,learning_rate)
    errors1[3] = NOR(epoch1,learning_rate)
   
    fig = plt.figure();
    axes = fig.add_subplot();
    axes.set_title(f'Error comparison for {epoch1} epochs with no hidden layer')
    axes.set_ylabel('MSE')
    axes.plot(errors1[0], label='AND')
    axes.plot(errors1[1], label='OR')
    axes.plot(errors1[2], label='NAND')
    axes.plot(errors1[3], label='NOR')
    fig.show()
    fig.legend()

    epoch2 = 10000     
         
    errors2 = np.zeros((2,epoch2))
    errors2[0] = XOR(epoch2,learning_rate)
    errors2[1] = XNOR(epoch2,learning_rate)
       
   
    fig1 = plt.figure();
    axes1 = fig1.add_subplot();
    axes1.set_title(f'Error comparison for {epoch2} epochs with 1 hidden layer')
    axes1.set_ylabel('MSE')
    axes1.plot(errors2[0], label='XOR')
    axes1.plot(errors2[1], label='XNOR')
    fig1.show()
    fig1.legend()
    
    
    
    plt.show()


compare()
