import numpy as np
import pytest

from src.evaluation import compute_metrics, predict_with_threshold, tune_threshold

CLASSES = np.array(["balance", "oos", "transfer"])
PROBS = np.array([
    [0.90, 0.05, 0.05],  # confident balance
    [0.40, 0.35, 0.25],  # unsure -> oos at tau 0.5
    [0.10, 0.80, 0.10],  # model itself predicts oos
    [0.20, 0.10, 0.70],  # transfer
])


def test_predict_with_threshold():
    assert predict_with_threshold(PROBS, CLASSES, 0.0).tolist() == ["balance", "balance", "oos", "transfer"]
    assert predict_with_threshold(PROBS, CLASSES, 0.5).tolist() == ["balance", "oos", "oos", "transfer"]


def test_compute_metrics():
    y_true = ["balance", "transfer", "oos", "oos", "transfer"]
    y_pred = ["balance", "oos", "oos", "balance", "transfer"]
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["in_scope_accuracy"] == pytest.approx(2 / 3)
    assert m["oos_recall"] == pytest.approx(1 / 2)


def test_tune_threshold_balances_in_scope_and_oos():
    y_true = ["balance", "oos", "oos", "transfer"]
    tau, m = tune_threshold(PROBS, CLASSES, y_true, grid=[0.0, 0.5, 0.95])
    assert tau == 0.5
    assert m["in_scope_accuracy"] == 1.0
    assert m["oos_recall"] == 1.0


def test_tune_threshold_prefers_lowest_tau_on_ties():
    y_true = ["balance", "balance", "oos", "transfer"]
    tau, _ = tune_threshold(PROBS, CLASSES, y_true, grid=[0.0, 0.3])
    assert tau == 0.0
