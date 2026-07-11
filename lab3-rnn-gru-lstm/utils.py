import numpy as np

def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred)**2))

def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return float(1 - ss_res / (ss_tot + 1e-8))

def train_test_split(X, y, test_ratio=0.2):
    n = len(X)
    idx = int(n * (1 - test_ratio))
    return X[:idx], X[idx:], y[:idx], y[idx:]
