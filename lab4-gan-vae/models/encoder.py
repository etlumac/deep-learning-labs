from nn.layers import Linear
from nn.activations import relu, drelu
import numpy as np

class Encoder:
    def __init__(self, x_dim=784, h=512, z_dim=32):
        self.l1 = Linear(x_dim, h)
        self.l2 = Linear(h, h//2)
        self.mu = Linear(h//2, z_dim)
        self.logvar = Linear(h//2, z_dim)

    def forward(self, x):
        self.h1 = relu(self.l1.forward(x))
        self.h2 = relu(self.l2.forward(self.h1))
        return self.mu.forward(self.h2), self.logvar.forward(self.h2)

    def backward(self, dmu, dlogvar):
        dh = self.mu.backward(dmu) + self.logvar.backward(dlogvar)
        dh = drelu(self.h2, dh)
        dh = self.l2.backward(dh)
        dh = drelu(self.h1, dh)
        self.l1.backward(dh)

    def step(self, lr):
        for l in [self.l1, self.l2, self.mu, self.logvar]:
            l.step(lr)
