"""Download CLINC150 (plus) and build the frozen train/val/test splits.

Every model in this project is trained on data/processed/train.csv, tuned on
val.csv and evaluated on test.csv, so this script is the single source of the
held-out test set.

Usage: python -m src.prepare_data
"""

import csv
import json
import random
import urllib.request
from collections import defaultdict

from src.paths import PROCESSED_DIR, RAW_DIR, ROOT

APPLIANCE_CSV = ROOT / "data" / "appliance_examples.csv"

# Pinned commit of https://github.com/clinc/oos-eval (CC BY 3.0).
CLINC_COMMIT = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
CLINC_BASE = f"https://raw.githubusercontent.com/clinc/oos-eval/{CLINC_COMMIT}/data"
CLINC_FILES = ("data_oos_plus.json", "domains.json")

APPLIANCE_DOMAIN = "appliance_support"
SEED = 42
SPLITS = ("train", "val", "test")


def download(name):
    path = RAW_DIR / name
    if not path.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{CLINC_BASE}/{name}", path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def intent_to_domain(domains):
    mapping = {intent: domain for domain, intents in domains.items() for intent in intents}
    mapping["oos"] = "oos"
    return mapping


def split_appliance(rows, seed, val_frac=0.2, test_frac=0.2):
    """Stratified split of (text, intent) rows: shuffled per intent with a fixed seed."""
    by_intent = defaultdict(list)
    for row in rows:
        by_intent[row[1]].append(row)
    rng = random.Random(seed)
    splits = {name: [] for name in SPLITS}
    for intent in sorted(by_intent):
        items = by_intent[intent][:]
        rng.shuffle(items)
        n_val = round(len(items) * val_frac)
        n_test = round(len(items) * test_frac)
        splits["val"] += items[:n_val]
        splits["test"] += items[n_val : n_val + n_test]
        splits["train"] += items[n_val + n_test :]
    return splits


def build_splits(clinc, domains, appliance_rows, seed):
    domain_of = intent_to_domain(domains)
    appliance = split_appliance(appliance_rows, seed)
    splits = {}
    for name in SPLITS:
        rows = [
            {"text": text, "intent": intent, "domain": domain_of[intent], "source": "clinc"}
            for text, intent in clinc[name] + clinc[f"oos_{name}"]
        ]
        rows += [
            {"text": text, "intent": intent, "domain": APPLIANCE_DOMAIN, "source": "appliance"}
            for text, intent in appliance[name]
        ]
        splits[name] = rows
    return splits


def load_appliance_rows():
    with open(APPLIANCE_CSV, newline="", encoding="utf-8") as f:
        return [(r["text"], r["intent"]) for r in csv.DictReader(f)]


def main():
    clinc = download("data_oos_plus.json")
    domains = download("domains.json")
    splits = build_splits(clinc, domains, load_appliance_rows(), SEED)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with open(PROCESSED_DIR / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "intent", "domain", "source"])
            writer.writeheader()
            writer.writerows(rows)

    intent_domain = {r["intent"]: r["domain"] for rows in splits.values() for r in rows}
    labels = {"intents": sorted(intent_domain), "intent_to_domain": dict(sorted(intent_domain.items()))}
    with open(PROCESSED_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    for name, rows in splits.items():
        n_oos = sum(r["intent"] == "oos" for r in rows)
        n_app = sum(r["source"] == "appliance" for r in rows)
        print(f"{name:5s} total={len(rows):6d} oos={n_oos:5d} appliance={n_app:3d}")
    print(f"intents (incl. oos): {len(labels['intents'])}")


if __name__ == "__main__":
    main()
