import numpy as np

# Object approach

np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

# init a layer 
# two ways
# trained model you saved
# you save the weights and the biases
# when initializaing load them
# And when saving just saving the weights and the biases
# 
# New neural netowrk
# weights initialized between -1 & 1
# so normalize and scale dataset
# biases tend to 0
###
class Layer_Dense:
    def __init__(self,n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# size of input -> how many feature in each sampple
# number of neurons whatever you want 
# layer 2 same size input as layer 1 output
layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
#print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)

