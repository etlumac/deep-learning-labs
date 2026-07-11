import numpy as np
import pandas as pd
from data_loader import load_data, preprocess_data
from model import init_model, forward_batch
from train import train, accuracy_from_preds
from metrics import (confusion_matrix_manual, precision_recall_f1_from_cm,
                     roc_auc_ovr, plot_confusion_matrix, plot_roc_classes,
                     save_classification_report, plot_tsne_analysis)
from config import *


def main():
    (x_train, y_train), (x_test, y_test) = load_data('3. mnist.npz')

    X_train, y_train, y_train_oh, X_val, y_val, y_val_oh = preprocess_data(
        x_train, x_test, y_train, y_test
    )

    #print(f"Train: {X_train.shape}, {y_train.shape}")
    #print(f"Val: {X_val.shape}, {y_val.shape}")

    # t-SNE анализ
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    plot_tsne_analysis(X_train_flat, y_train)

    # Инициализация и обучение модели
    model = init_model(seed=1)

    model, history = train(
        model,
        X_train, y_train_oh,
        X_val, y_val_oh,
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE
    )

    # Оценка модели
    probs_val, _ = forward_batch(X_val, model)
    preds_val = np.argmax(probs_val, axis=1)
    y_val_true = np.argmax(y_val_oh, axis=1)

    # Метрики
    cm = confusion_matrix_manual(y_val_true, preds_val)
    precision, recall, f1 = precision_recall_f1_from_cm(cm)
    aucs, rocs = roc_auc_ovr(probs_val, y_val_oh)

    # Визуализации
    plot_confusion_matrix(cm, filename='confusion.png')
    plot_roc_classes(rocs, aucs, classes=list(range(10)), filename='roc.png')

    # Сохранение результатов
    save_classification_report(precision, recall, f1, aucs)

    df_cm = pd.DataFrame(cm, index=range(10), columns=range(10))
    df_cm.to_csv("confusion_matrix.csv")

    print("Val accuracy:", accuracy_from_preds(preds_val, y_val_true))
    print("Per-class precision:", precision)
    print("Per-class recall:", recall)
    print("Per-class f1:", f1)
    print("AUCs:", aucs)
    print("Saved confusion_matrix.csv and classification_report.csv")


if __name__ == "__main__":
    main()