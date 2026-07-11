import numpy as np


class LSTM:
    def __init__(self, input_size, hidden_size, output_size, seed=1):
        rng = np.random.RandomState(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # input
        self.Wi = rng.randn(hidden_size, input_size) * 0.1
        self.Ui = rng.randn(hidden_size, hidden_size) * 0.1
        self.bi = np.zeros(hidden_size)
        # forget
        self.Wf = rng.randn(hidden_size, input_size) * 0.1
        self.Uf = rng.randn(hidden_size, hidden_size) * 0.1
        self.bf = np.zeros(hidden_size)
        # output
        self.Wo = rng.randn(hidden_size, input_size) * 0.1
        self.Uo = rng.randn(hidden_size, hidden_size) * 0.1
        self.bo = np.zeros(hidden_size)
        # cell candidate
        self.Wc = rng.randn(hidden_size, input_size) * 0.1
        self.Uc = rng.randn(hidden_size, hidden_size) * 0.1
        self.bc = np.zeros(hidden_size)
        # readout
        self.Why = rng.randn(output_size, hidden_size) * 0.1
        self.by = np.zeros(output_size)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, x):
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)

        self.xs = []
        self.is_ = []
        self.fs_ = []
        self.os_ = []
        self.gs_ = []
        self.cs = []
        self.hs = []

        for t in range(x.shape[0]):
            xt = x[t]
            self.xs.append(xt)
            i = self.sigmoid(self.Wi.dot(xt) + self.Ui.dot(h) + self.bi)
            f = self.sigmoid(self.Wf.dot(xt) + self.Uf.dot(h) + self.bf)
            o = self.sigmoid(self.Wo.dot(xt) + self.Uo.dot(h) + self.bo)
            g = np.tanh(self.Wc.dot(xt) + self.Uc.dot(h) + self.bc)

            c = f * c + i * g
            h = o * np.tanh(c)

            self.is_.append(i)
            self.fs_.append(f)
            self.os_.append(o)
            self.gs_.append(g)
            self.cs.append(c)
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
        self.dWi = np.zeros_like(self.Wi)
        self.dUi = np.zeros_like(self.Ui)
        self.dbi = np.zeros_like(self.bi)
        self.dWf = np.zeros_like(self.Wf)
        self.dUf = np.zeros_like(self.Uf)
        self.dbf = np.zeros_like(self.bf)
        self.dWo = np.zeros_like(self.Wo)
        self.dUo = np.zeros_like(self.Uo)
        self.dbo = np.zeros_like(self.bo)
        self.dWc = np.zeros_like(self.Wc)
        self.dUc = np.zeros_like(self.Uc)
        self.dbc = np.zeros_like(self.bc)
        self.dWhy = np.zeros_like(self.Why)
        self.dby = np.zeros_like(self.by)

    def bptt(self, x, target, learning_rate=1e-3, grad_clip=1.0):
        y_pred = self.forward(x)
        err = y_pred - target
        loss = 0.5 * (err ** 2).sum()

        self.zero_grads()
        self.dWhy += np.outer(err, self.h_last)
        self.dby += err

        dh_next = self.Why.T.dot(err)
        dc_next = np.zeros(self.hidden_size)

        for t in reversed(range(len(self.xs))):
            x_t = self.xs[t]
            i_t = self.is_[t]
            f_t = self.fs_[t]
            o_t = self.os_[t]
            g_t = self.gs_[t]
            c_t = self.cs[t]
            h_t = self.hs[t]
            c_prev = self.cs[t - 1] if t > 0 else np.zeros_like(c_t)
            h_prev = self.hs[t - 1] if t > 0 else np.zeros_like(h_t)

            #  h_t = o_t * tanh(c_t)
            dh = dh_next
            do = dh * np.tanh(c_t)
            d_o_input = do * o_t * (1 - o_t)

            dc = dh * o_t * (1 - np.tanh(c_t) ** 2) + dc_next

            # filters
            di = dc * g_t
            df = dc * c_prev
            dg = dc * i_t

            d_i_input = di * i_t * (1 - i_t)
            d_f_input = df * f_t * (1 - f_t)
            d_g_input = dg * (1 - g_t ** 2)

            self.dWi += np.outer(d_i_input, x_t)
            self.dUi += np.outer(d_i_input, h_prev)
            self.dbi += d_i_input

            self.dWf += np.outer(d_f_input, x_t)
            self.dUf += np.outer(d_f_input, h_prev)
            self.dbf += d_f_input

            self.dWo += np.outer(d_o_input, x_t)
            self.dUo += np.outer(d_o_input, h_prev)
            self.dbo += d_o_input

            self.dWc += np.outer(d_g_input, x_t)
            self.dUc += np.outer(d_g_input, h_prev)
            self.dbc += d_g_input

            dh_prev = (self.Ui.T.dot(d_i_input) +
                       self.Uf.T.dot(d_f_input) +
                       self.Uo.T.dot(d_o_input) +
                       self.Uc.T.dot(d_g_input))

            dc_prev = dc * f_t

            dh_next = dh_prev
            dc_next = dc_prev


        for g in [self.dWi, self.dUi, self.dbi, self.dWf, self.dUf, self.dbf,
                  self.dWo, self.dUo, self.dbo, self.dWc, self.dUc, self.dbc,
                  self.dWhy, self.dby]:
            np.clip(g, -grad_clip, grad_clip, out=g)


        self.Wi -= learning_rate * self.dWi
        self.Ui -= learning_rate * self.dUi
        self.bi -= learning_rate * self.dbi

        self.Wf -= learning_rate * self.dWf
        self.Uf -= learning_rate * self.dUf
        self.bf -= learning_rate * self.dbf

        self.Wo -= learning_rate * self.dWo
        self.Uo -= learning_rate * self.dUo
        self.bo -= learning_rate * self.dbo

        self.Wc -= learning_rate * self.dWc
        self.Uc -= learning_rate * self.dUc
        self.bc -= learning_rate * self.dbc

        self.Why -= learning_rate * self.dWhy
        self.by -= learning_rate * self.dby

        return float(loss)
