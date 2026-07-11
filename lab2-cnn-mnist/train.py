import numpy as np
from config import *
from model import forward_batch, backward_batch

def cross_entropy_loss(probs, targets):
    eps = 1e-12
    return -np.mean(np.sum(targets * np.log(probs + eps), axis=1))


def train(model, X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
          target_loss=TARGET_LOSS):

    n = X_train.shape[0]
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        idx = np.random.permutation(n)
        X_train, y_train = X_train[idx], y_train[idx]
        train_losses = []

        for i in range(0, n, batch_size):
            xb, yb = X_train[i:i + batch_size], y_train[i:i + batch_size]
            probs, cache = forward_batch(xb, model)
            loss = cross_entropy_loss(probs, yb)
            train_losses.append(loss)

            dW_conv, db_conv, dW_fc, db_fc = backward_batch(probs, yb, cache, model)
            model['W_conv'] -= lr * dW_conv
            model['b_conv'] -= lr * db_conv
            model['W_fc'] -= lr * dW_fc
            model['b_fc'] -= lr * db_fc.squeeze()

        train_loss = np.mean(train_losses)
        probs_val, _ = forward_batch(X_val, model)
        val_loss = cross_entropy_loss(probs_val, y_val)
        val_preds = np.argmax(probs_val, axis=1)
        val_acc = accuracy_from_preds(val_preds, np.argmax(y_val, axis=1))

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs} - loss {train_loss:.4f} - val_loss {val_loss:.4f} - val_acc {val_acc:.4f}")

        if val_loss < target_loss:
            break

    return model, history


def accuracy_from_preds(preds, labels):
    return np.mean(preds == labels)