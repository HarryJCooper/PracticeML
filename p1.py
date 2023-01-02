# THIS IS A SINGLE NEURON, EVERY INPUT HAS A WEIGHT, BUT ONLY ONE BIAS/NEURON

# THE INPUT LAYER IS JUST THE VALUES YOU PUT IN

# THE OUTPUT NEURONS 

# IN MATHEMATICS A LIST IS A VECTOR
# A LIST OF LISTS IS A 2D ARRAY
# A MATRIX IS JUST A RECTANGULAR ARRAY
# A LIST OF VECTORS IS A MATRIX
# A LIST OF LISTS OF LISTS IS A 3D ARRAY
# A TENSOR IS AN OBJECT THAT CAN BE REPRESENTED AS AN ARRAY

# the dot_product e.g a . b = [1, 2, 3] . [2, 3, 4] = 1 * 2 + 2 * 3 + 3 * 4 = 20
# the dot_product of two vectors (lists) results in a single scalar value
# a scaler value is a value that only has one component to it

# ACTIVATION FUNCTION IS REQUIRED TO STOP IT ALL BEING LINEAR, ACTIVATION REFERS TO A CERTAIN CHARGE BEING NEEDED TO ACTIVATE A SYNAPSE

import numpy as np
import nnfs

nnfs.init()

np.random.seed(0)
# WANT A TO PASS A BATCH OF INPUT SAMPLES
X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

for i in inputs:
    output.append(max(0, 1))
    

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons) # PARAMETERS ARE THE SHAPE
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)

# weights = [[0.2, 0.8, -0.5, 1.0],
#         [0.5, -0.91, 0.26, -0.5],
#         [-0.26, -0.27, 0.17, 0.87]]

# biases = [2, 3, 0.5]

# weights2 = [[0.1, -0.14, 0.5],
#         [-0.5, 0.12, -0.33],
#         [-0.44, 0.73, -0.13]]

# biases2 = [-1, 2, -0.5]

# # THE FIRST ELEMENT YOU PASS THE np.dot DETERMINES THE OUTPUT
# # THE FIRST ELEMENT CAN BE A MATRIX (A LIST/ARRAY OF VECTORS)
# # THEN IT ITERATES THROUGH EACH VECTOR AND GETS THE DOT PRODUCT AGAINST THE INPUT

# layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
# layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

# print(layer2_outputs)
