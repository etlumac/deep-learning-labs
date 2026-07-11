from nn.layers import Linear
from nn.activations import relu, drelu, sigmoid, dsigmoid

class Discriminator:
    def __init__(self, x_dim=784, h=256):
        self.l1 = Linear(x_dim, h)
        self.l2 = Linear(h, h//2)
        self.l3 = Linear(h//2, 1)

    def forward(self, x):
        self.h1 = relu(self.l1.forward(x))
        self.h2 = relu(self.l2.forward(self.h1))
        self.logit = self.l3.forward(self.h2)
        self.prob = sigmoid(self.logit)
        return self.prob

    def backward(self, grad):
        grad = dsigmoid(self.prob, grad)
        grad = self.l3.backward(grad)
        grad = drelu(self.h2, grad)
        grad = self.l2.backward(grad)
        grad = drelu(self.h1, grad)
        self.l1.backward(grad)

    def step(self, lr):
        for l in [self.l1, self.l2, self.l3]:
            l.step(lr)
