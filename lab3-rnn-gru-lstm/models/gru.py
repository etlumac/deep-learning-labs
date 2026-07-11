import numpy as np

class GRU:
    def __init__(self, input_size, hidden_size, output_size, seed=123):
        rng = np.random.RandomState(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # update
        self.Wz = rng.randn(hidden_size, input_size) * 0.1
        self.Uz = rng.randn(hidden_size, hidden_size) * 0.1
        self.bz = np.zeros(hidden_size)
        # reset
        self.Wr = rng.randn(hidden_size, input_size) * 0.1
        self.Ur = rng.randn(hidden_size, hidden_size) * 0.1
        self.br = np.zeros(hidden_size)
        # new candidate
        self.Wh = rng.randn(hidden_size, input_size) * 0.1
        self.Uh = rng.randn(hidden_size, hidden_size) * 0.1
        self.bh = np.zeros(hidden_size)
        # readout
        self.Why = rng.randn(output_size, hidden_size) * 0.1
        self.by = np.zeros(output_size)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, x):
        h = np.zeros(self.hidden_size)
        self.xs = []
        self.zs = []
        self.rs = []
        self.h_tildes = []
        self.hs = []

        for t in range(x.shape[0]):
            xt = x[t]
            self.xs.append(xt)
            z = self.sigmoid(self.Wz.dot(xt) + self.Uz.dot(h) + self.bz)
            r = self.sigmoid(self.Wr.dot(xt) + self.Ur.dot(h) + self.br)
            h_tilde = np.tanh(self.Wh.dot(xt) + self.Uh.dot(r * h) + self.bh)
            h = (1 - z) * h + z * h_tilde

            self.zs.append(z)
            self.rs.append(r)
            self.h_tildes.append(h_tilde)
            self.hs.append(h)

        self.h_last = h
        y = self.Why.dot(h) + self.by
        return y

    def predict_batch(self, X):
        preds = []
        for i in range(len(X)):
            preds.append(self.forward(X[i]))
        return np.vstack(preds).squeeze()

    def zero_grads(self):
        self.dWz = np.zeros_like(self.Wz)
        self.dUz = np.zeros_like(self.Uz)
        self.dbz = np.zeros_like(self.bz)
        self.dWr = np.zeros_like(self.Wr)
        self.dUr = np.zeros_like(self.Ur)
        self.dbr = np.zeros_like(self.br)
        self.dWh = np.zeros_like(self.Wh)
        self.dUh = np.zeros_like(self.Uh)
        self.dbh = np.zeros_like(self.bh)
        self.dWhy = np.zeros_like(self.Why)
        self.dby = np.zeros_like(self.by)

    def bptt(self, x, target, learning_rate=1e-3, grad_clip=1.0):
        y_pred = self.forward(x)
        err = y_pred - target
        loss = 0.5 * (err**2).sum()


        self.zero_grads()

        self.dWhy += np.outer(err, self.h_last)
        self.dby += err


        dh_next = self.Why.T.dot(err)

        # BPTT
        for t in reversed(range(len(self.xs))):
            x_t = self.xs[t]
            z_t = self.zs[t]
            r_t = self.rs[t]
            h_tilde_t = self.h_tildes[t]
            h_t = self.hs[t]
            h_prev = self.hs[t-1] if t > 0 else np.zeros_like(h_t)

            dh = dh_next

            #  h_t = (1 - z_t)*h_prev + z_t*h_tilde_t
            dh_tilde = dh * z_t
            dz = dh * (h_tilde_t - h_prev)
            dh_prev_from_direct = dh * (1 - z_t)

            # h_tilde = tanh(a), a = Wh x_t + Uh (r_t * h_prev) + bh
            da = dh_tilde * (1 - h_tilde_t**2)
            self.dWh += np.outer(da, x_t)
            self.dUh += np.outer(da, r_t * h_prev)
            self.dbh += da

            d_rh = self.Uh.T.dot(da)
            dr = d_rh * h_prev
            dh_prev_from_h_tilde = d_rh * r_t

            dz_input = dz * z_t * (1 - z_t)
            dr_input = dr * r_t * (1 - r_t)

            self.dWz += np.outer(dz_input, x_t)
            self.dUz += np.outer(dz_input, h_prev)
            self.dbz += dz_input

            self.dWr += np.outer(dr_input, x_t)
            self.dUr += np.outer(dr_input, h_prev)
            self.dbr += dr_input

            dh_prev_from_gates = self.Uz.T.dot(dz_input) + self.Ur.T.dot(dr_input)

            # total dh_prev for previous time-step
            dh_next = dh_prev_from_direct + dh_prev_from_h_tilde + dh_prev_from_gates


        for g in [self.dWz, self.dUz, self.dbz, self.dWr, self.dUr, self.dbr,
                  self.dWh, self.dUh, self.dbh, self.dWhy, self.dby]:
            np.clip(g, -grad_clip, grad_clip, out=g)


        self.Wz -= learning_rate * self.dWz
        self.Uz -= learning_rate * self.dUz
        self.bz -= learning_rate * self.dbz

        self.Wr -= learning_rate * self.dWr
        self.Ur -= learning_rate * self.dUr
        self.br -= learning_rate * self.dbr

        self.Wh -= learning_rate * self.dWh
        self.Uh -= learning_rate * self.dUh
        self.bh -= learning_rate * self.dbh

        self.Why -= learning_rate * self.dWhy
        self.by -= learning_rate * self.dby

        return float(loss)
