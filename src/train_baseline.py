"""Classical baseline: tf-idf (word 1-2 grams + char 2-5 grams) + logistic regression.

C and the out-of-scope threshold are chosen on the validation set. The test set
is not used here.

Usage: python -m src.train_baseline
"""

import time

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

from src.data import load_split
from src.evaluation import tune_threshold
from src.paths import MODELS_DIR

MODEL_PATH = MODELS_DIR / "baseline.joblib"
C_GRID = (1.0, 10.0, 100.0)


def build_pipeline(C):
    features = make_union(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2),
    )
    # Balanced weights: each appliance intent has 12 training examples vs ~100 per CLINC intent.
    return make_pipeline(features, LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))


def main():
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    print(f"train={len(X_train)} val={len(X_val)}")

    best = None
    for C in C_GRID:
        start = time.perf_counter()
        model = build_pipeline(C).fit(X_train, y_train)
        tau, appliance_tau, m = tune_threshold(model.predict_proba(X_val), model.classes_, y_val)
        score = (m["in_scope_accuracy"] + m["oos_recall"]) / 2
        print(
            f"C={C:<6g} tau={tau:.2f} appliance_tau={appliance_tau:.2f} val_acc={m['accuracy']:.4f} "
            f"in_scope={m['in_scope_accuracy']:.4f} oos_recall={m['oos_recall']:.4f} "
            f"({time.perf_counter() - start:.0f}s)",
            flush=True,
        )
        if best is None or score > best[0]:
            best = (score, C, tau, appliance_tau, model)

    _, C, tau, appliance_tau, model = best
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "tau": tau, "appliance_tau": appliance_tau, "C": C}, MODEL_PATH)
    print(f"saved {MODEL_PATH.name}: C={C:g} tau={tau:.2f} appliance_tau={appliance_tau:.2f} "
          f"size={MODEL_PATH.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
