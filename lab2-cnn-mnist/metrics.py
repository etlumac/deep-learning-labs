import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE


def confusion_matrix_manual(y_true, y_pred, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def precision_recall_f1_from_cm(cm):
    tp = np.diag(cm).astype(float)
    fp = np.sum(cm, axis=0).astype(float) - tp
    fn = np.sum(cm, axis=1).astype(float) - tp
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where(tp + fp == 0, 0.0, tp / (tp + fp))
        recall = np.where(tp + fn == 0, 0.0, tp / (tp + fn))
        f1 = np.where(precision + recall == 0, 0.0, 2 * precision * recall / (precision + recall))
    return precision, recall, f1


def roc_auc_ovr(probs, y_true_onehot, n_points=200):
    n_classes = probs.shape[1]
    aucs = np.zeros(n_classes, dtype=float)
    rocs = {}

    for c in range(n_classes):
        scores = probs[:, c]
        labels = y_true_onehot[:, c].astype(int)
        thresh = np.linspace(1.0, 0.0, n_points)
        tprs = []
        fprs = []
        P = max(1, np.sum(labels == 1))
        N = max(1, np.sum(labels == 0))

        for t in thresh:
            preds = (scores >= t).astype(int)
            tp = np.sum((preds == 1) & (labels == 1))
            fp = np.sum((preds == 1) & (labels == 0))
            tpr = tp / P
            fpr = fp / N
            tprs.append(tpr)
            fprs.append(fpr)

        fprs = np.array(fprs)
        tprs = np.array(tprs)
        order = np.argsort(fprs)
        f_sorted = fprs[order]
        t_sorted = tprs[order]
        auc = np.trapezoid(t_sorted, f_sorted)
        aucs[c] = auc
        rocs[c] = (f_sorted, t_sorted)

    return aucs, rocs


def plot_confusion_matrix(cm, labels=None, figsize=(6, 6), filename=None):
    if labels is None:
        labels = list(range(cm.shape[0]))

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
    plt.show()
    plt.close()


def plot_roc_classes(rocs, aucs, classes=None, filename=None):
    plt.figure(figsize=(6, 6))
    if classes is None:
        classes = sorted(rocs.keys())

    for c in classes:
        fpr, tpr = rocs[c]
        plt.plot(fpr, tpr, label=f'class {c} (AUC={aucs[c]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.6)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curves")
    plt.legend(loc='lower right', fontsize='small')
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150)
    plt.show()
    plt.close()


def save_classification_report(precision, recall, f1, aucs, filename="classification_report.csv"):
    df = pd.DataFrame({
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": aucs
    })
    df.index.name = "class"
    df.to_csv(filename)
    print(f"Saved {filename}")


def plot_tsne_analysis(X, y_true, perplexity=10, learning_rate=200, max_iter=1000):
    try:
        import plotly.express as px
        has_plotly = True
    except ImportError:
        has_plotly = False

    embed = TSNE(n_components=2, perplexity=perplexity,
                 learning_rate=learning_rate, max_iter=max_iter)
    X_embedded = embed.fit_transform(X)
    print('Расхождение Кульбака-Лейблера после оптимизации: ', embed.kl_divergence_)

    if has_plotly:
        fig = px.scatter(None, x=X_embedded[:, 0], y=X_embedded[:, 1],
                         labels={"x": "Dimension 1", "y": "Dimension 2"},
                         opacity=1, color=y_true.astype(str))
        fig.update_layout(dict(plot_bgcolor='white'))
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                         zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                         showline=True, linewidth=1, linecolor='black')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                         zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                         showline=True, linewidth=1, linecolor='black')
        fig.update_layout(title_text="t-SNE")
        fig.update_traces(marker=dict(size=3))
        fig.show()
    else:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_true, cmap='tab10', s=3)
        plt.colorbar(scatter)
        plt.title("t-SNE Visualization")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.show()