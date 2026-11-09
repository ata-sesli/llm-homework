# Customer Support Ticket Router

SWE015 homework, Use Case 2. A free-text support message is normalized, classified into an intent, and routed to a team. Low-confidence and out-of-scope messages go to a human agent, and small talk gets a rule-based reply.

Status: data preparation, preprocessing, small-talk rules, the classical baseline and the feedforward network are done. The Transformer, the quantized model and the app are not built yet.

## Setup

Requires Python 3.13. gensim has no Python 3.14 wheels yet.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Storage location (optional)

Downloaded data, processed splits, trained models, gensim word vectors (about 1 GB) and the Hugging Face cache all go under one storage root. By default that is the repo folder. To use another disk, set `TICKET_ROUTER_STORAGE`, either as an environment variable or in a `.env` file in the repo root (git-ignored):

```bash
echo "TICKET_ROUTER_STORAGE=/Volumes/external/ticket-router" > .env
```

## Prepare the data

```bash
python -m src.prepare_data
```

This downloads CLINC150 from a pinned commit into `<storage>/data/raw/`, adds the hand-written appliance examples, and writes the frozen splits to `<storage>/data/processed/`:

| Split | Rows | Out-of-scope | Appliance |
|---|---|---|---|
| train | 15,310 | 250 | 60 |
| val | 3,120 | 100 | 20 |
| test | 5,520 | 1,000 | 20 |

There are 156 labels: 150 CLINC intents, 5 appliance intents and `oos`. The routed team is the intent's domain: the 10 CLINC domains plus `appliance_support`. The intent-to-domain map is in `<storage>/data/processed/labels.json`.

Every model is trained on `train.csv`, tuned on `val.csv`, and evaluated once on `test.csv`. The rerun is deterministic: seed 42 and a pinned source commit.

The official CLINC splits already contain 5 duplicate texts across train/val/test, some with different labels. They are kept so the test set stays identical to the standard benchmark.

## Preprocessing and rules

`src/preprocess.py` (`preprocess(text, known_words)`) is shared by every model and the app:

- regex extraction of model codes with a product prefix (`DW-6534`, `wm4410` → `WM-4410`), removed from the text
- NFKC, lowercasing, Turkish character folding, collapsing repeated letters, removing punctuation
- Turkish product names mapped to English (`bulaşık makinem` → `dishwasher`)
- misspelled product names corrected by Levenshtein distance against a product lexicon (`dishwaser` → `dishwasher`). Tokens are left alone if they appear in the CLINC training vocabulary, are inflected product words, or don't share the candidate's first letter.

`src/rules.py` holds ELIZA-style regex rules for greetings, thanks, goodbyes and "are you a bot". Each rule must match the whole message, and captured text is pronoun-reflected. On the full dataset the rules fire only on CLINC small-talk intents. When a classifier predicts a small-talk intent, `reply_for_intent()` gives a canned reply.

Known failure: the name "Owen" in one test message is corrected to "oven".

## Evaluation rule

`src/evaluation.py` applies the same rule to every model. A message is out-of-scope if the model's top class is `oos` or its top probability is below a threshold τ. τ is tuned on the validation set only, maximizing the mean of in-scope accuracy and OOS recall. Reported metrics: accuracy (all 156 labels), in-scope accuracy and OOS recall.

## Models

### 1. Classical baseline

```bash
python -m src.train_baseline
```

- Features: tf-idf over word 1–2-grams plus character 2–5-grams (`char_wb`).
- Classifier: logistic regression with balanced class weights.
- C is chosen from {1, 10, 100} by the validation score.
- Saved to `<storage>/models/baseline.joblib`.

Balanced weights are used because each appliance intent has 12 training examples against about 100 per CLINC intent.

### 2. Feedforward network on fastText vectors

```bash
python -m src.train_ffn
```

- Input: the mean of the unit-length word vectors of a message.
- Network: Linear(300→512) → ReLU → Dropout(0.3) → Linear(512→156) → softmax.
- Training: mini-batch SGD (lr 0.1, momentum 0.9, batch 64) with backpropagation, on a class-weighted cross-entropy loss, for up to 150 epochs. The epoch with the best validation score is kept.

Two word-vector sources, both through gensim, are compared on validation:

- pretrained English fastText wiki-news vectors (`fasttext-wiki-news-subwords-300` from gensim-data), limited to the 100k most frequent words. Unknown words are skipped.
- a FastText model trained on our training messages (100-dimensional, with vectors for unknown words built from character n-grams).

The pretrained vectors won. The model is saved to `<storage>/models/ffn.pt`, with its vectors in `ffn_vectors.kv`.

The subword `.bin` version of the pretrained vectors is a 4.5 GB download and was not used.

## Tests

```bash
python -m pytest -q
```

## Data sources

- **CLINC150 (`data_oos_plus.json`, `domains.json`):** Larson et al., 2019, *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*, EMNLP-IJCNLP. https://github.com/clinc/oos-eval (CC BY 3.0), commit `828f809`.
- **`data/appliance_examples.csv`:** 100 home-appliance support messages over 5 intents (`appliance_repair`, `appliance_spare_parts`, `appliance_warranty`, `appliance_installation`, `appliance_usage`), 20 each, split 60/20/20 per intent. The first draft was written with an AI assistant; see the AI usage statement.
