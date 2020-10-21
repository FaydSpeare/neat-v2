import numpy as np

def sigmoid(x):
    x = np.maximum(-100., np.minimum(100., 5. * x))
    return 1. / (1. + np.exp(-x))

def sin(x):
    return np.sin(x)

def gauss(x):
    x = np.maximum(-3.4, np.minimum(3.4, x))
    return np.exp(-5.0 * x ** 2)

def identity(x):
    return x

def relu(x):
    return np.maximum(x, 0.)

def abs(x):
    return np.abs(x)