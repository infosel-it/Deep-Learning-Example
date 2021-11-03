import numpy as np

input_features = np.array([[0,0], [0,1], [1,0],[1,1]])
print(input_features.shape) 
print(input_features)

target_output =np.array([[0,1,1,1]])
target_output = target_output.reshape(4,1)
print(target_output.shape) 

weights = np.array([[0.1], [0.2]])
print(weights.shape)

# Derivative of Sigmoid Function

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid(x))