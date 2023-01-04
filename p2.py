# Considering single node
# 3 inputs 3 weights and a bias

# inputs can be from input layer or
# outputs from neurons 

# The neruon can also be an output 
# so only adds one more input and weight

'''

inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

output = inputs[0]*weights[0] + \
    inputs[1]*weights[1] + \
    inputs[2]*weights[2] + \
    inputs[3]*weights[3] + bias


print(output)
#4.8
'''

# For ouptut layer
# need 3 weight sets fro 3 neurons

# We tweak weights and biasis to get best array of outputs
inputs = [1, 2, 3, 2.5]

weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91,0.26,-0.5]
weights3 = [-0.26,-0.27,0.17,0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5 

output = [inputs[0]*weights1[0] + \
    inputs[1]*weights1[1] + \
    inputs[2]*weights1[2] + \
    inputs[3]*weights1[3] + bias1,
    
    inputs[0]*weights2[0] + \
    inputs[1]*weights2[1] + \
    inputs[2]*weights2[2] + \
    inputs[3]*weights2[3] + bias2,
    
    inputs[0]*weights3[0] + \
    inputs[1]*weights3[1] + \
    inputs[2]*weights3[2] + \
    inputs[3]*weights3[3] + bias3,
    
    ]


print(output)