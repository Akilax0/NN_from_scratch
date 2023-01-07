# Softmax 
# input -> exponentiate -> normalize -> output 
# input -> Softmax -> output 

# These are output values
# predicting largest 
# index = 0 

# But when training model need to know how wrong an output is  
# and values unbounded

import math 
import numpy as np
import nnfs 

# https://github.com/Sentdex/nnfs/blob/master/nnfs/core.py
nnfs.init()
# layer_outputs = [4.8, 1.21, 2.385]

# need batch of inputs to get batch of outputs
# numpy calculates for batch by default

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

# layer_outputs = [4.8, 4.79, 4.25]

# Hence need new activation function 
# Idealy to get a probability distribution as output
# ReLu clips the negatives

# But if linear still need to optimize 
# Need to find direction 

# Expeonential funciton 
# solves negative problem 
# accounts for them without tossing out the meaning of a negative number value 

# Finding exponents
exp_values = np.exp(layer_outputs)

# However exponents can blow up 
# to solve this we take the max value
# and min all values from it 
# thus making the max a 1 and the rest negative 
# must be done before the exponents 
# overflow error handling

'''
E = math.e
exp_values = []

for output in layer_outputs:
    exp_values.append(E**output)

print(exp_values)
'''

# Next step normalize
# Single output / Sum of all outputs 

# axis = 0 for cols
# axis = 1 for rows
# keepdims 
norm_values = exp_values / np.sum(layer_outputs, axis=1, keepdims=True)

'''
norm_base = sum(exp_values)
norm_values = []

norm_values = exp_values / np.sum(exp_values)

for value in exp_values:
    norm_values.append(value / norm_base) 
'''
print(norm_values)
