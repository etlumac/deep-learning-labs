import numpy as np

def relu(x):
    return np.maximum(0, x)

def drelu(x, grad):
    return grad * (x > 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y, grad):
    return grad * y * (1 - y)
