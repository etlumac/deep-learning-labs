import numpy as np

def glorot_init(in_dim, out_dim):
    limit = np.sqrt(6 / (in_dim + out_dim))
    return np.random.uniform(-limit, limit, (in_dim, out_dim)).astype(np.float32)

class Linear:
    def __init__(self, in_dim, out_dim):
        self.W = glorot_init(in_dim, out_dim)
        self.b = np.zeros((1, out_dim), dtype=np.float32)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.dW = self.x.T @ grad / self.x.shape[0]
        self.db = grad.mean(axis=0, keepdims=True)
        return grad @ self.W.T

    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db
