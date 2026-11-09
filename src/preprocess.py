"""Text normalization shared by every model and by the app.

preprocess() runs, in order:
1. regex extraction (and removal) of product model codes such as DW-6534;
2. Unicode NFKC, lowercasing and Turkish character folding (ç→c, ı→i, ...);
3. collapsing characters repeated 3+ times to two ("pleeease" → "pleease");
4. removing punctuation;
5. mapping a few Turkish product names to English ("bulasik makinem" → "dishwasher");
6. correcting misspelled product names by minimum edit distance against a lexicon.
   A token is only corrected if it is not a known word, not an inflected product
   word, and shares its first letter with the candidate.
"""

import csv
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from src.paths import PROCESSED_DIR

TRAIN_CSV = PROCESSED_DIR / "train.csv"

# Model codes are a product prefix + 3-5 digits (DW-6534, wm4410). Limiting the
# prefixes keeps flight numbers like DL123 in CLINC text from matching.
PRODUCT_CODE_PREFIXES = {
    "DW": "dishwasher", "WM": "washing machine", "FR": "refrigerator", "FZ": "freezer",
    "OV": "oven", "DR": "dryer", "MW": "microwave", "CT": "cooktop",
}
MODEL_CODE_RE = re.compile(rf"\b({'|'.join(PRODUCT_CODE_PREFIXES)})-?(\d{{3,5}})\b", re.IGNORECASE)
REPEAT_RE = re.compile(r"(.)\1{2,}")
NON_WORD_RE = re.compile(r"[^a-z0-9' ]+")
TURKISH_FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")

TURKISH_PRODUCTS = [
    (re.compile(r"\bbulasik makin\w*"), "dishwasher"),
    (re.compile(r"\bcamasir makin\w*"), "washing machine"),
    (re.compile(r"\bkurutma makin\w*"), "dryer"),
    (re.compile(r"\bbuzdolab\w*"), "refrigerator"),
    (re.compile(r"\bdondurucu\w*"), "freezer"),
    (re.compile(r"\bmikrodalga\w*"), "microwave"),
    (re.compile(r"\bfirin\w*"), "oven"),
    (re.compile(r"\bocak\w*"), "cooktop"),
]

PRODUCT_LEXICON = (
    "appliance", "cooktop", "dishwasher", "dryer", "freezer", "fridge", "machine",
    "microwave", "oven", "refrigerator", "stove", "washer", "washing",
)
MIN_CORRECTION_LEN = 4
INFLECTION_SUFFIXES = ("ing", "ed", "es", "s")


@dataclass
class Preprocessed:
    text: str
    model_codes: list = field(default_factory=list)
    corrections: list = field(default_factory=list)


def levenshtein(a, b):
    """Minimum edit distance with unit costs for insertion, deletion and substitution."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def extract_model_codes(text):
    return [f"{letters.upper()}-{digits}" for letters, digits in MODEL_CODE_RE.findall(text)]


def max_distance(word):
    return 1 if len(word) <= 7 else 2


def is_inflected_product(token):
    """True for forms like "cooktops", "washed", "refrigerated" whose stem starts a product word."""
    return any(
        token.endswith(suffix) and any(word.startswith(token[: -len(suffix)]) for word in PRODUCT_LEXICON)
        for suffix in INFLECTION_SUFFIXES
    )


def correct_token(token, known_words):
    if (
        len(token) < MIN_CORRECTION_LEN
        or token in PRODUCT_LEXICON
        or token in known_words
        or is_inflected_product(token)
    ):
        return token
    limit = max_distance(token)
    candidates = (
        word for word in PRODUCT_LEXICON
        if word[0] == token[0] and abs(len(word) - len(token)) <= limit
    )
    best = min(((levenshtein(token, word), word) for word in candidates), default=None)
    return best[1] if best and best[0] <= limit else token


def preprocess(text, known_words=frozenset()):
    codes = extract_model_codes(text)
    text = MODEL_CODE_RE.sub(" ", text)
    text = unicodedata.normalize("NFKC", text).replace("İ", "i").lower().translate(TURKISH_FOLD)
    text = REPEAT_RE.sub(r"\1\1", text)
    text = NON_WORD_RE.sub(" ", text)
    text = " ".join(text.split())
    for pattern, english in TURKISH_PRODUCTS:
        text = pattern.sub(english, text)

    tokens, corrections = [], []
    for token in text.split():
        fixed = correct_token(token, known_words)
        if fixed != token:
            corrections.append((token, fixed))
        tokens.append(fixed)
    return Preprocessed(" ".join(tokens), codes, corrections)


def load_known_words(path=TRAIN_CSV, min_count=1):
    """Tokens seen at least min_count times in the CLINC training text.

    Used so that ordinary words close to a product name ("over" ~ "oven") are not
    "corrected". Appliance rows are excluded because they contain deliberate typos.
    """
    counts = Counter()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["source"] == "clinc":
                counts.update(NON_WORD_RE.sub(" ", row["text"].lower()).split())
    return frozenset(word for word, n in counts.items() if n >= min_count)
