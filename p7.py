# Calculating Loss with Categorical Cross-Entropy
#     
# Metric for error
# If only cared about accuracy 
# Make attempts to optimiz using binary wont work
# need a confidence score - probability
# it would be helpful as the optimizer to adjust the weights and the 
# biases
# Mean absolute error - loss funcion (regression a value than a probability)
# as more closer to the value less error

# Determining how wrong the value
# For classification using softmx -> Categorical Cross-Entropy
# 
# One hot encoding
# Have a vector of n classes long 
# Have all zeros except at the target class where is 1
# Classes : 3
# Label : 1
# One-hot : [0, 1, 0]

# Natural log ln(x)
# base: Euler's number e 
# log generally solving x for 
# e**x = b 


# Example:
# Classes : 3
# Label: 0
# One-hot: [1, 0, 0]
# Prediction: [0.7, 0.1, 0.2]
# L = - sigma(ylog(ypred)) = - ((1.log(0.7)) + (0.log(0.1)) + 0.(log(0.2)) )
#   = 0.35667494393873245



import numpy as np
import nnfs
import cd

nnfs.init()

class Layer_Dense:
    def __init__(self,n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

class Activation_Softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

X,y = cd.create_data(points=100,classes=3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])
