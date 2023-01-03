# Considering single node
# 3 inputs 3 weights and a bias

# inputs can be from input layer or
# outputs from neurons 

# The neruon can also be an output 
# so only adds one more niput and weight

inputs = [1, 2, 3]
weights = [0.2, 0.8, -0.5]
bias = 2

output = inputs[0]*weights[0] + \
    inputs[1]*weights[1] + \
    inputs[2]*weights[2] + bias


print(output)