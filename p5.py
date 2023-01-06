# Activation Functions !!

# 1. Step Function 
# y = 1 x > 0
# y = 1 x <= 0

# Applies after neuron 
# after all the calculations
# makes the output either 1 or 0

# Every neuron will have a activation function 
# Generally output layer has different

# 2. Signmoid 
# Easier and reliable to train 
# As before after due to granuality
# y = 1/(1+e-x)
# we didnt know how close to the ends 
# hence important in loss 

# 3. ReLu
# y = x x> 0
# y = 0 x<= 0
# As before again after calculations
# Vanishing gradient problem in Signmoid
# Fast -> simple calculation 
# works 


# Why Activation function 
# with linear function can only fit linear data
# ReLu almost linear 
# we can strengthten by weights
# offset point of activation by bias 
# 
# more neurons better fit

import numpy as np
import nnfs

# random seed init
# same data type numpy
nnfs.init()

# np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:
    def __init__(self,n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Actiation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
#print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)





