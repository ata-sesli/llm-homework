"""Evaluate every model once on the held-out test set and benchmark size and latency.

Metrics use the thresholds tuned on validation; nothing is tuned here.
Latency is per message (batch size 1) on CPU, from normalized text to class
probabilities, including each model's own featurization (tf-idf, word vectors or
tokenizer). The shared preprocessing step is the same for every model and excluded.

Usage: python -m src.evaluate   (writes results.csv)
"""

import csv
import platform
import random
import subprocess
import time

from src.models import load_models, size_mb  # imported first: sets HF_HOME

import numpy as np
import onnxruntime
import torch

from src.data import load_split
from src.evaluation import compute_metrics
from src.paths import ROOT

RESULTS_CSV = ROOT / "results.csv"
LATENCY_RUNS = 200
LATENCY_WARMUP = 20
SEED = 42


def evaluate_model(model, texts, labels):
    labels = np.asarray(labels)
    pred = model.predict(texts)
    metrics = compute_metrics(labels, pred)
    appliance = np.char.startswith(labels.astype(str), "appliance_")
    metrics["appliance_accuracy"] = float((pred[appliance] == labels[appliance]).mean())
    return metrics


def measure_latency(predict_proba, texts, runs=LATENCY_RUNS, warmup=LATENCY_WARMUP):
    """Call predict_proba on one message at a time; the first `warmup` calls are not timed."""
    for i in range(warmup):
        predict_proba([texts[i % len(texts)]])
    times = []
    for i in range(runs):
        start = time.perf_counter()
        predict_proba([texts[i % len(texts)]])
        times.append((time.perf_counter() - start) * 1000)
    return {"mean_ms": float(np.mean(times)), "p50_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)), "runs": runs}


def hardware():
    def sysctl(key):
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
    if platform.system() == "Darwin":
        cpu, ram = sysctl("machdep.cpu.brand_string"), int(sysctl("hw.memsize")) / 2**30
        return f"{cpu}, {ram:.0f} GB RAM, macOS {platform.mac_ver()[0]}, CPU only"
    return f"{platform.processor() or platform.machine()}, {platform.system()}, CPU only"


def main():
    X_test, y_test = load_split("test")
    X_val, _ = load_split("val")  # latency inputs: unlabeled validation messages
    latency_texts = random.Random(SEED).sample(X_val, LATENCY_RUNS)
    hw = hardware()
    print(f"test={len(X_test)} hardware: {hw}; torch threads={torch.get_num_threads()} "
          f"onnxruntime {onnxruntime.__version__}", flush=True)

    rows = []
    for key, model in load_models().items():
        start = time.perf_counter()
        m = evaluate_model(model, X_test, y_test)
        lat = measure_latency(model.predict_proba, latency_texts)
        rows.append({"model": model.name, "accuracy": m["accuracy"], "oos_recall": m["oos_recall"],
                     "in_scope_accuracy": m["in_scope_accuracy"], "appliance_accuracy": m["appliance_accuracy"],
                     "size_mb": size_mb(model.files), "latency_mean_ms": lat["mean_ms"],
                     "latency_p50_ms": lat["p50_ms"], "latency_p95_ms": lat["p95_ms"], "latency_runs": lat["runs"],
                     "hardware": hw})
        r = rows[-1]
        print(f"{model.name}: acc={r['accuracy']:.4f} oos_recall={r['oos_recall']:.4f} "
              f"in_scope={r['in_scope_accuracy']:.4f} appliance={r['appliance_accuracy']:.2f} "
              f"size={r['size_mb']:.1f} MB latency={r['latency_mean_ms']:.2f} ms "
              f"(p95 {r['latency_p95_ms']:.2f}) ({time.perf_counter() - start:.0f}s)", flush=True)

    with RESULTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items()})
    print(f"wrote {RESULTS_CSV.name}")


if __name__ == "__main__":
    main()
