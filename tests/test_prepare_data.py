import csv
from collections import Counter
from pathlib import Path

from src.prepare_data import (
    APPLIANCE_CSV,
    build_splits,
    intent_to_domain,
    split_appliance,
)

DOMAINS = {"banking": ["balance", "transfer"], "small_talk": ["greeting"]}
CLINC = {
    "train": [("what's my balance", "balance"), ("hi", "greeting")],
    "oos_train": [("how tall is everest", "oos")],
    "val": [("send money", "transfer")],
    "oos_val": [("sing a song", "oos")],
    "test": [("hello there", "greeting")],
    "oos_test": [("who won the game", "oos")],
}


def appliance_rows(per_intent=20):
    return [
        (f"{intent} message {i}", intent)
        for intent in ("appliance_repair", "appliance_warranty")
        for i in range(per_intent)
    ]


def test_intent_to_domain_adds_oos():
    assert intent_to_domain(DOMAINS) == {
        "balance": "banking",
        "transfer": "banking",
        "greeting": "small_talk",
        "oos": "oos",
    }


def test_split_appliance_is_stratified_60_20_20():
    splits = split_appliance(appliance_rows(), seed=0)
    for name, expected in (("train", 12), ("val", 4), ("test", 4)):
        counts = Counter(intent for _, intent in splits[name])
        assert counts == {"appliance_repair": expected, "appliance_warranty": expected}


def test_split_appliance_is_deterministic_and_disjoint():
    a = split_appliance(appliance_rows(), seed=0)
    b = split_appliance(appliance_rows(), seed=0)
    assert a == b
    texts = [t for name in ("train", "val", "test") for t, _ in a[name]]
    assert len(texts) == len(set(texts)) == 40


def test_build_splits_merges_oos_and_appliance():
    splits = build_splits(CLINC, DOMAINS, appliance_rows(), seed=0)
    train = splits["train"]
    assert len(train) == 2 + 1 + 24
    oos = [r for r in train if r["intent"] == "oos"]
    assert oos == [
        {"text": "how tall is everest", "intent": "oos", "domain": "oos", "source": "clinc"}
    ]
    appliance = [r for r in train if r["source"] == "appliance"]
    assert {r["domain"] for r in appliance} == {"appliance_support"}
    assert len(splits["test"]) == 1 + 1 + 8


def test_appliance_csv_has_20_unique_examples_per_intent():
    with open(APPLIANCE_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    counts = Counter(r["intent"] for r in rows)
    assert len(counts) == 5
    assert set(counts.values()) == {20}
    texts = [r["text"].strip().lower() for r in rows]
    assert len(texts) == len(set(texts))
    assert all(r["intent"].startswith("appliance_") for r in rows)
