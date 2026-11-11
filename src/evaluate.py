# evaluate.py - metrics with uncertainty (use on the TEST set once, after all decisions)
import numpy as np
from sklearn.metrics import f1_score, recall_score, confusion_matrix, classification_report


def predict_all(model, ds):
    """Returns y_true, y_pred, probs for a (images, labels) tf.data.Dataset."""
    y, p = [], []
    for xb, yb in ds:
        p.append(model.predict(xb, verbose=0))
        y.append(yb.numpy())
    probs = np.concatenate(p)
    return np.concatenate(y), probs.argmax(1), probs


def bootstrap_ci(y_true, y_pred, metric, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap over test images. Returns (point, low, high)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    vals = [metric(y_true[b], y_pred[b])
            for b in (rng.choice(idx, size=len(idx), replace=True) for _ in range(n_boot))]
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return metric(y_true, y_pred), lo, hi


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def per_class_recall(y_true, y_pred, class_names):
    r = recall_score(y_true, y_pred, labels=range(len(class_names)), average=None, zero_division=0)
    return dict(zip(class_names, np.round(r, 3)))


def report(y_true, y_pred, class_names):
    print(classification_report(y_true, y_pred, labels=range(len(class_names)),
                                target_names=class_names, zero_division=0))
    print(confusion_matrix(y_true, y_pred, labels=range(len(class_names))))
