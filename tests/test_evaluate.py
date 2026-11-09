import numpy as np
import pytest

from src.evaluate import evaluate_model, measure_latency
from src.models import RouterModel, size_mb

CLASSES = np.array(["appliance_repair", "balance", "oos"])


def fake_model(probs_by_text, tau=0.5, appliance_tau=0.3):
    def predict_proba(texts):
        return np.array([probs_by_text[t] for t in texts])
    return RouterModel("fake", CLASSES, tau, appliance_tau, predict_proba, files=[])


def test_router_model_predict_uses_both_thresholds():
    model = fake_model({"oven": [0.4, 0.3, 0.3], "money": [0.3, 0.4, 0.3], "hi": [0.1, 0.1, 0.8]})
    assert model.predict(["oven", "money", "hi"]).tolist() == ["appliance_repair", "oos", "oos"]


def test_evaluate_model_reports_task_metrics():
    model = fake_model({"oven": [0.4, 0.3, 0.3], "money": [0.1, 0.8, 0.1], "weird": [0.2, 0.7, 0.1]})
    m = evaluate_model(model, ["oven", "money", "weird"], ["appliance_repair", "balance", "oos"])
    assert m["accuracy"] == pytest.approx(2 / 3)
    assert m["in_scope_accuracy"] == 1.0
    assert m["oos_recall"] == 0.0
    assert m["appliance_accuracy"] == 1.0


def test_measure_latency_times_one_message_per_call():
    calls = []
    stats = measure_latency(lambda batch: calls.append(list(batch)), ["a", "b", "c"], runs=7, warmup=2)
    assert len(calls) == 9 and all(len(batch) == 1 for batch in calls)
    assert [b[0] for b in calls[2:]] == ["a", "b", "c", "a", "b", "c", "a"]
    assert stats["runs"] == 7
    assert 0 <= stats["p50_ms"] <= stats["p95_ms"]
    assert stats["mean_ms"] >= 0


def test_size_mb_counts_files_and_directories(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 1_000_000)
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "w.bin").write_bytes(b"x" * 500_000)
    assert size_mb([tmp_path / "a.bin", folder]) == pytest.approx(1.5)
