import numpy as np

def bce(x, y, eps=1e-8):
    return -np.mean(y*np.log(x+eps) + (1-y)*np.log(1-x+eps))

def dbce(x, y, eps=1e-8):
    # Derivative of sum of per-sample BCE wrt probability x (no 1/N scaling).
    # Layers already average gradients over the batch.
    return (-y / (x + eps)) + ((1 - y) / (1 - x + eps))

def kl_div(mu, logvar):
    return 0.5 * np.mean(np.sum(np.exp(logvar) + mu**2 - 1 - logvar, axis=1))
