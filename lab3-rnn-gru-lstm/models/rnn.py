import numpy as np

class VanillaRNN:

    def __init__(self, input_size, hidden_size, output_size, seed=42):
        rng = np.random.RandomState(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # params
        self.Wxh = rng.randn(hidden_size, input_size) * 0.1
        self.Whh = rng.randn(hidden_size, hidden_size) * 0.1
        self.bh = np.zeros(hidden_size)
        self.Why = rng.randn(output_size, hidden_size) * 0.1
        self.by = np.zeros(output_size)


    def forward(self, x):
        h_prev = np.zeros(self.hidden_size)
        self.hs = []
        for t in range(x.shape[0]):
            xt = x[t]
            h = np.tanh(self.Wxh.dot(xt) + self.Whh.dot(h_prev) + self.bh)
            self.hs.append(h)
            h_prev = h
        self.h_last = h_prev
        y = self.Why.dot(self.h_last) + self.by
        return y

    def predict_batch(self, X):
        preds = []
        for i in range(len(X)):
            preds.append(self.forward(X[i]))
        return np.vstack(preds).squeeze()

    def zero_grads(self):
        self.dWxh = np.zeros_like(self.Wxh)
        self.dWhh = np.zeros_like(self.Whh)
        self.dbh = np.zeros_like(self.bh)
        self.dWhy = np.zeros_like(self.Why)
        self.dby = np.zeros_like(self.by)

    def bptt(self, x, target, learning_rate=1e-3, grad_clip=1.0):
        y_pred = self.forward(x)
        err = y_pred - target

        self.zero_grads()
        self.dWhy += np.outer(err, self.h_last)
        self.dby += err
        # backprop into last hidden
        dh = self.Why.T.dot(err)
        # BPTT
        for t in reversed(range(len(self.hs))):
            h = self.hs[t]
            prev_h = self.hs[t-1] if t > 0 else np.zeros_like(h)
            dtanh = (1 - h**2) * dh
            self.dbh += dtanh
            self.dWxh += np.outer(dtanh, x[t])
            self.dWhh += np.outer(dtanh, prev_h)
            dh = self.Whh.T.dot(dtanh)

        for g in [self.dWxh, self.dWhh, self.dbh, self.dWhy, self.dby]:
            np.clip(g, -grad_clip, grad_clip, out=g)

        self.Wxh -= learning_rate * self.dWxh
        self.Whh -= learning_rate * self.dWhh
        self.bh  -= learning_rate * self.dbh
        self.Why -= learning_rate * self.dWhy
        self.by  -= learning_rate * self.dby
        loss = 0.5 * (err**2).sum()
        return float(loss)
