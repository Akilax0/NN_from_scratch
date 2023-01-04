inputs = [1, 2, 3, 2.5]

# weights and biases 
# by optimizer
# weight magnitude
# bias offset

weights = [[0.2, 0.8, -0.5, 1.0], \
    [0.5, -0.91,0.26,-0.5], \
    [-0.26,-0.27,0.17,0.87]]

biases = [2,3,0.5]


'''
output = []
for i in range(len(weights)):
    temp = 0
    for j in  range(len(inputs)):
        temp = temp + inputs[j] * weights[i][j]
    
    temp = temp + biases[i]
    output.append(temp) 

print(output)
'''

layer_outputs = [] # Output of current layer
for neuron_weights, neuron_bias in zip(weights,biases):
    neuron_output = 0 # Output of given neuron
    for n_input, weight in zip(inputs,neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)
print(layer_outputs)


# SHape is dimentsion 
# Array l = [1,5,6,2]   |  Shape : (4,) | Type 1D array, Vector
# Array lol= [[1,5,6,2],[3,2,1,3]] | Shape : (2,4) | 2D Array Matrix
# Array lolol = [[[11,5,6,2],[3,2,1,3]],[[5,2,1,2],[6,4,8,4]],[[2,8,5,3],[1,1,9,4]]]
# Shape: (3,2,4) | Type 3D array

# TEnsor - is an object that can be represented as an array



