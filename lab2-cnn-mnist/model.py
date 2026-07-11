import numpy as np
from config import *

def init_model(seed=SEED):
    rng = np.random.RandomState(seed)
    model = {}
    model['W_conv'] = rng.randn(3,3,1,8).astype(np.float32) * np.sqrt(2./(3*3*1))
    model['b_conv'] = np.zeros((8,), dtype=np.float32)
    flat = 13*13*8
    model['W_fc'] = rng.randn(flat, 10).astype(np.float32) * np.sqrt(2./flat)
    model['b_fc'] = np.zeros((1,10), dtype=np.float32)
    return model

def conv_forward(x, w, b):
    N, H, W, C_in = x.shape
    kh, kw, _, C_out = w.shape
    H_out = H - kh + 1
    W_out = W - kw + 1
    out = np.zeros((N, H_out, W_out, C_out), dtype=np.float32)
    for n in range(N):
        for i in range(H_out):
            for j in range(W_out):
                patch = x[n, i:i+kh, j:j+kw, :]
                for f in range(C_out):
                    out[n, i, j, f] = np.sum(patch * w[:, :, :, f]) + b[f]
    cache = (x, w, b)
    return out, cache

def conv_backward(dout, cache):
    x, w, b = cache
    N, H, W, C_in = x.shape
    kh, kw, _, C_out = w.shape

    dW = np.zeros_like(w)
    db = np.zeros_like(b)
    dx = np.zeros_like(x)

    H_out, W_out = H - kh + 1, W - kw + 1

    for n in range(N):
        for i in range(H_out):
            for j in range(W_out):
                for f in range(C_out):
                    db[f] += dout[n, i, j, f]
                    patch = x[n, i:i+kh, j:j+kw, :]
                    dW[:, :, :, f] += patch * dout[n, i, j, f]
                    dx[n, i:i+kh, j:j+kw, :] += w[:, :, :, f] * dout[n, i, j, f]

    return dx, dW, db


def relu_forward(x):
    out = np.maximum(0, x)
    return out, x

def relu_backward(dout, cache):
    x = cache
    return dout * (x > 0).astype(np.float32)

def maxpool_forward(x, ph=2, pw=2):
    N, H, W, C = x.shape
    H_out, W_out = H // ph, W // pw
    out = np.zeros((N, H_out, W_out, C), dtype=np.float32)
    mask = {}
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h0, w0 = i*ph, j*pw
                    patch = x[n, h0:h0+ph, w0:w0+pw, c]
                    out[n, i, j, c] = np.max(patch)
                    mask[(n, c, i, j)] = np.argmax(patch)
    cache = (x.shape, ph, pw, mask)
    return out, cache

def maxpool_backward(dout, cache):
    x_shape, ph, pw, mask = cache
    N, H, W, C = x_shape
    dx = np.zeros(x_shape, dtype=np.float32)
    H_out, W_out = H // ph, W // pw
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h0, w0 = i*ph, j*pw
                    max_pos = mask[(n, c, i, j)]
                    r, s = divmod(max_pos, pw)
                    dx[n, h0+r, w0+s, c] = dout[n, i, j, c]
    return dx

def dense_forward(x, W, b):
    out = x.dot(W) + b
    return out, (x, W, b)

def dense_backward(dout, cache):
    x, W, b = cache
    dW = x.T.dot(dout)
    db = np.sum(dout, axis=0, keepdims=True)
    dx = dout.dot(W.T)
    return dx, dW, db

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def forward_batch(x, model):
    conv_out, conv_cache = conv_forward(x, model['W_conv'], model['b_conv'])
    relu_out, relu_cache = relu_forward(conv_out)
    pool_out, pool_cache = maxpool_forward(relu_out, 2, 2)
    flat = pool_out.reshape(x.shape[0], -1)
    fc_out, fc_cache = dense_forward(flat, model['W_fc'], model['b_fc'])
    probs = softmax(fc_out)
    return probs, (conv_cache, relu_cache, pool_cache, fc_cache, pool_out.shape)

def backward_batch(probs, y_true, cache, model):
    N = probs.shape[0]
    dlogits = (probs - y_true) / N

    conv_cache, relu_cache, pool_cache, fc_cache, pool_shape = cache

    dx_flat, dW_fc, db_fc = dense_backward(dlogits, fc_cache)
    dx_pool = dx_flat.reshape(pool_shape)

    d_relu = maxpool_backward(dx_pool, pool_cache)

    d_conv = relu_backward(d_relu, relu_cache)

    dx, dW_conv, db_conv = conv_backward(d_conv, conv_cache)

    return dW_conv, db_conv, dW_fc, db_fc