import numpy as np

def square(size):
    return np.array([(i / (size - 1), j / (size - 1)) for i in range(size) for j in range(size)]).astype('float32')

def rectangle(shape):
    isize, jsize = shape
    return np.array([(2*(i / max(1, isize - 1))-1, 2*(j / max(1, jsize - 1))-1) for i in range(isize) for j in range(jsize)]).astype('float32')

