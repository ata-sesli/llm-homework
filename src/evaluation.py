"""Out-of-scope decision rule and metrics shared by every model.

A message is predicted out-of-scope ("oos") if the model's top class is "oos" or
its top probability is below a threshold. The threshold is tau, except when the
top class is an appliance intent: then it is appliance_tau. The appliance intents
have about 12 training examples each against about 100 per CLINC intent, so the
models are less confident on them, and a single tau sent most of them to "oos".

Both thresholds are tuned on the validation set only, maximizing the mean of
in-scope accuracy and OOS recall. tau is tuned first, exactly as if it were the
only threshold (ties go to the lowest tau). appliance_tau is tuned next with tau
fixed, so predictions for every other class do not change (ties go to the highest,
i.e. strictest, appliance_tau, because the validation set has few appliance rows).
"""

import numpy as np

OOS = "oos"
APPLIANCE_PREFIX = "appliance_"
DEFAULT_GRID = np.round(np.arange(0.0, 1.0, 0.05), 2)


def predict_with_threshold(probs, classes, tau, appliance_tau=None):
    classes = np.asarray(classes)
    top = probs.argmax(axis=1)
    pred = classes[top].astype(object)
    if appliance_tau is None:
        appliance_tau = tau
    is_appliance = np.char.startswith(classes.astype(str), APPLIANCE_PREFIX)[top]
    pred[probs.max(axis=1) < np.where(is_appliance, appliance_tau, tau)] = OOS
    return pred


def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    in_scope = y_true != OOS
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "in_scope_accuracy": float((y_true[in_scope] == y_pred[in_scope]).mean()),
        "oos_recall": float((y_pred[~in_scope] == OOS).mean()),
    }


def _score(metrics):
    return (metrics["in_scope_accuracy"] + metrics["oos_recall"]) / 2


def tune_threshold(probs, classes, y_true, grid=DEFAULT_GRID):
    """Return (tau, appliance_tau, metrics) for the best balanced score."""
    best_tau = None
    for tau in grid:
        score = _score(compute_metrics(y_true, predict_with_threshold(probs, classes, tau)))
        if best_tau is None or score > best_tau[0]:
            best_tau = (score, float(tau))
    tau = best_tau[1]

    best = None
    for appliance_tau in grid:
        m = compute_metrics(y_true, predict_with_threshold(probs, classes, tau, appliance_tau))
        if best is None or _score(m) >= best[0]:
            best = (_score(m), float(appliance_tau), m)
    return tau, best[1], best[2]
