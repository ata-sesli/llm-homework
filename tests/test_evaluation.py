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
    tau, appliance_tau, m = tune_threshold(PROBS, CLASSES, y_true, grid=[0.0, 0.5, 0.95])
    assert tau == 0.5
    assert appliance_tau == 0.95  # no appliance class: every value ties, the strictest is kept
    assert m["in_scope_accuracy"] == 1.0
    assert m["oos_recall"] == 1.0


def test_tune_threshold_prefers_lowest_tau_on_ties():
    y_true = ["balance", "balance", "oos", "transfer"]
    tau, _, _ = tune_threshold(PROBS, CLASSES, y_true, grid=[0.0, 0.3])
    assert tau == 0.0


APP_CLASSES = np.array(["appliance_repair", "balance", "oos"])
APP_PROBS = np.array([
    [0.45, 0.30, 0.25],  # unsure appliance message
    [0.40, 0.10, 0.50],  # out-of-scope by argmax
    [0.10, 0.85, 0.05],  # confident balance
    [0.05, 0.55, 0.40],  # unsure balance
    [0.05, 0.50, 0.45],  # unsure balance
    [0.05, 0.90, 0.05],  # confident balance
])


def test_appliance_tau_applies_only_to_appliance_predictions():
    pred = predict_with_threshold(APP_PROBS, APP_CLASSES, 0.6, appliance_tau=0.4)
    assert pred.tolist() == ["appliance_repair", "oos", "balance", "oos", "oos", "balance"]
    assert predict_with_threshold(APP_PROBS, APP_CLASSES, 0.6).tolist() == [
        "oos", "oos", "balance", "oos", "oos", "balance"]


def test_tune_threshold_tunes_appliance_tau_after_the_global_tau():
    y_true = ["appliance_repair", "oos", "balance", "oos", "oos", "balance"]
    tau, appliance_tau, m = tune_threshold(APP_PROBS, APP_CLASSES, y_true, grid=[0.0, 0.4, 0.6, 0.9])
    assert tau == 0.6  # unchanged by the appliance rows: chosen as if there were one threshold
    assert appliance_tau == 0.4  # the strictest value that still accepts the appliance message
    assert m["in_scope_accuracy"] == 1.0 and m["oos_recall"] == 1.0
