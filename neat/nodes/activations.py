import math

def sigmoid(x):
    x = max(-100., min(100., 5. * x))
    return 1. / (1. + math.exp(-x))

def sin(x):
    return math.sin(x)

def gauss(x):
    x = max(-3.4, min(3.4, x))
    return math.exp(-5.0 * x ** 2)

def identity(x):
    return x

def relu(x):
    return max(x, 0.)