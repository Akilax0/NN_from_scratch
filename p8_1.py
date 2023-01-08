# softmax_outputs = [[0.7, 0.1, 0.2],
#                   [0.1, 0.5, 0.4],
#                   [0.02, 0.9, 0.08]]

# Classes:
# 0- dog
# 1 - cat
# 2 - human

# class_targets = [dog, cat, cat] = [0, 1, 1]
#  class_targets = [0,
#                   1,
#                   1]

# For confidences get at relevent indexes
# 0.7 
# 0.5
# 0.9 

import numpy as np
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])

class_targets = [0, 1, 1]

# numpy array can be indexed as such
# first dimension indices, second dimension indices
# like a 2D array but cleaner
print(softmax_outputs[[0, 1, 2], class_targets])
print(softmax_outputs[range(len(softmax_outputs)), class_targets])
neg_loss = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
average_loss = np.mean(neg_loss)
print(average_loss)


