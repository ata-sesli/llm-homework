"""Loading the frozen splits written by src.prepare_data."""

import csv

from src.paths import PROCESSED_DIR
from src.preprocess import load_known_words, preprocess


def load_split(name):
    """Return (normalized_texts, intents) for "train", "val" or "test"."""
    known_words = load_known_words()
    with open(PROCESSED_DIR / f"{name}.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    texts = [preprocess(r["text"], known_words).text for r in rows]
    return texts, [r["intent"] for r in rows]
