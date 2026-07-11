from nn.layers import Linear
from nn.activations import relu, drelu, sigmoid, dsigmoid

class Decoder:
    def __init__(self, z_dim=32, h=256, x_dim=784):
        self.l1 = Linear(z_dim, h)
        self.l2 = Linear(h, h)
        self.l3 = Linear(h, x_dim)

    def forward(self, z):
        self.h1 = relu(self.l1.forward(z))
        self.h2 = relu(self.l2.forward(self.h1))
        self.logits = self.l3.forward(self.h2)
        self.out = sigmoid(self.logits)
        return self.out

    def backward(self, grad):
        grad = dsigmoid(self.out, grad)
        grad = self.l3.backward(grad)
        grad = drelu(self.h2, grad)
        grad = self.l2.backward(grad)
        grad = drelu(self.h1, grad)
        grad = self.l1.backward(grad)
        return grad

    def step(self, lr):
        for l in [self.l1, self.l2, self.l3]:
            l.step(lr)

