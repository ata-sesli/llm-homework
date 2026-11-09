"""Out-of-scope decision rule and metrics shared by every model.

A message is predicted out-of-scope ("oos") if the model's top class is "oos" or
its top probability is below a threshold tau. tau is tuned on the validation set
only, by maximizing the mean of in-scope accuracy and OOS recall.
"""

import numpy as np

OOS = "oos"
DEFAULT_GRID = np.round(np.arange(0.0, 1.0, 0.05), 2)


def predict_with_threshold(probs, classes, tau):
    classes = np.asarray(classes)
    pred = classes[probs.argmax(axis=1)].astype(object)
    pred[probs.max(axis=1) < tau] = OOS
    return pred


def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    in_scope = y_true != OOS
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "in_scope_accuracy": float((y_true[in_scope] == y_pred[in_scope]).mean()),
        "oos_recall": float((y_pred[~in_scope] == OOS).mean()),
    }


def tune_threshold(probs, classes, y_true, grid=DEFAULT_GRID):
    """Return (tau, metrics) for the tau with the best balanced score; ties go to the lowest tau."""
    best = None
    for tau in grid:
        m = compute_metrics(y_true, predict_with_threshold(probs, classes, tau))
        score = (m["in_scope_accuracy"] + m["oos_recall"]) / 2
        if best is None or score > best[0]:
            best = (score, float(tau), m)
    return best[1], best[2]
