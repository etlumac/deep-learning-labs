from config import *
import numpy as np


def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)


def preprocess_data(x_train, x_test, y_train, y_test):
    # Нормализация
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # Добавляем канал
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Разделение 70/30
    X_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)

    N = X_all.shape[0]
    idx = np.random.RandomState(0).permutation(N)
    split = int(TRAIN_TEST_SPLIT * N)

    X_train_final, y_train_final = X_all[idx[:split]], y_all[idx[:split]]
    X_val_final, y_val_final = X_all[idx[split:]], y_all[idx[split:]]
    y_train_oh_final = one_hot(y_train_final, NUM_CLASSES)
    y_val_oh_final = one_hot(y_val_final, NUM_CLASSES)

    return (X_train_final, y_train_final, y_train_oh_final,
            X_val_final, y_val_final, y_val_oh_final)


def one_hot(labels, n_classes=10):
    y = np.zeros((labels.size, n_classes), dtype=np.float32)
    y[np.arange(labels.size), labels] = 1.0
    return y