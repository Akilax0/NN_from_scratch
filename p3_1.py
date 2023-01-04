# Dot product : element wise multiply and add them 

import numpy as np

inputs = [1,2,3,2.5]
weights = [[0.2, 0.8, -0.5, 1.0], \
    [0.5, -0.91,0.26,-0.5], \
    [-0.26,-0.27,0.17,0.87]]

biases = [2,3,0.5]

# np.dot(a,b) 
# a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
output = np.dot(weights,inputs) + biases
print(output)

# matrix hence first element 
# defiens how return is indexed
# 3 sets of weights henc need to include as first 
# on dot product