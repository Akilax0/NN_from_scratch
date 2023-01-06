import numpy as np

np.random.seed(0)


def create_data(points, classes):
    X = np.zeros((points*classes,2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np
