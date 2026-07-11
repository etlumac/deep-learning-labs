import argparse
import numpy as np
import time
from data_loader import load_and_prepare
from utils import mse, rmse, r2_score, train_test_split
from models.rnn import VanillaRNN
from models.gru import GRU
from models.lstm import LSTM
import matplotlib.pyplot as plt
import os

plt.rcParams['figure.figsize'] = (10,4)

def train(model_name='rnn', epochs=5, seq_len=24, lr=1e-3, hidden=32, test_ratio=0.2):
    X, y, meta = load_and_prepare(seq_len=seq_len)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=test_ratio)
    input_size = X.shape[2]

    if model_name == 'rnn':
        model = VanillaRNN(input_size, hidden, 1)
    elif model_name == 'gru':
        model = GRU(input_size, hidden, 1)
    elif model_name == 'lstm':
        model = LSTM(input_size, hidden, 1)
    else:
        raise ValueError('model must be rnn|gru|lstm')

    losses = []
    start = time.time()
    for ep in range(epochs):
        ep_loss = 0.0
        for i in range(len(X_train)):
            x = X_train[i]
            t = y_train[i]
            l = model.bptt(x, t, learning_rate=lr)
            ep_loss += float(l)
        avg = ep_loss / len(X_train)
        losses.append(avg)
        print(f'[Epoch {ep+1}/{epochs}] train loss: {avg:.6f}')
    print('Training completed in {:.1f}s'.format(time.time() - start))

    y_pred = model.predict_batch(X_test)
    mse_val = mse(y_test, y_pred)
    rmse_val = rmse(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print('Test MSE:', mse_val, 'RMSE:', rmse_val, 'R2:', r2)

    with open('results/metrics.txt', 'w') as f:
        f.write(f'Model: {model_name}\n')
        f.write(f'MSE: {mse_val}\nRMSE: {rmse_val}\nR2: {r2}\n')

    os.makedirs('results', exist_ok=True)
    plt.figure()
    plt.plot(losses, marker='o')
    plt.title('Train loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.savefig('results/train_loss.png')
    plt.close()

    plt.figure()
    Nplot = min(200, len(y_test))
    plt.plot(y_test[:Nplot], label='true')
    plt.plot(y_pred[:Nplot], label='pred')
    plt.legend()
    plt.title('Pred vs True (first {} test samples)'.format(Nplot))
    plt.savefig('results/pred_vs_true.png')
    plt.close()

    print('Saved: results/train_loss.png, results/pred_vs_true.png, metrics.txt')
    return model, meta, (mse_val, rmse_val, r2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='rnn')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seq_len', type=int, default=24)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=int, default=32)
    args = parser.parse_args()
    train(model_name=args.model, epochs=args.epochs, seq_len=args.seq_len, lr=args.lr, hidden=args.hidden)
