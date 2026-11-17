# Customer Support Ticket Router

SWE015 homework, Use Case 2. A free-text support message is normalized, classified into an intent, and routed to a team. Low-confidence and out-of-scope messages go to a human agent, and small talk gets a rule-based reply.

Status: data preparation, preprocessing, small-talk rules, the classical baseline, the feedforward network, the BERT teacher, the distilled, quantized student, and the test-set evaluation with benchmarks, and the Gradio app are done.

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

`src/evaluation.py` applies the same rule to every model. A message is out-of-scope if the model's top class is `oos` or its top probability is below a threshold. The threshold is τ, or a separate τ_appliance when the top class is an appliance intent. Reported metrics: accuracy (all 156 labels), in-scope accuracy and OOS recall.

Both thresholds are tuned on the validation set only, maximizing the mean of in-scope accuracy and OOS recall:

1. τ is tuned first, as if it were the only threshold.
2. τ_appliance is tuned next with τ fixed, so predictions for the other 151 classes don't change. Ties go to the strictest value.

Why a second threshold: each appliance intent has 12 training examples against about 100 per CLINC intent, so every model is less confident on them. In the first training run, with a single τ, the quantized student sent 15 of the 20 appliance validation messages to `oos`, although its top class was correct on 18. The 20 appliance rows are too few for the balanced score to notice. Before adopting the rule, it was checked with 2-fold cross-validation inside the validation set. Across all four models, held-out appliance accuracy went up and the held-out balanced score stayed the same within noise. A threshold per domain (11 thresholds) was also tried; it overfit the validation set, losing 3–5 points of held-out score.

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

### 3. BERT teacher

```bash
python -m src.train_bert
```

- `bert-base-uncased` with a linear classification head on the `[CLS]` token, fine-tuned end to end.
- Training: AdamW (lr 5e-5, weight decay 0.01), linear warmup over the first 10% of steps then linear decay, batch 32, 5 epochs, class-weighted cross-entropy.
- Runs on Apple-silicon GPU (MPS) when available, else CUDA, else CPU.
- The epoch with the best validation score is kept, along with its τ.
- Saved to `<storage>/models/bert_teacher/` with `meta.json` (τ, classes, validation metrics).

### 4. Distilled student

```bash
python -m src.distill
```

- Student: `distilbert-base-uncased` (6 layers instead of 12), with a new classification head.
- The teacher's logits on the training set are computed once. The student is trained with the same optimizer, schedule, batch size and epoch count as the teacher, on the loss α·T²·KL(teacher_T ‖ student_T) + (1−α)·class-weighted cross-entropy, with α = 0.5 and temperature T = 2.
- The epoch with the best validation score is kept, along with its τ.
- Saved to `<storage>/models/bert_student/`.

### 5. Quantized student

```bash
python -m src.quantize
```

- The student is exported to ONNX with dynamic batch size and sequence length, then quantized with onnxruntime dynamic quantization: weights of the linear layers are stored as 8-bit integers, and activations are quantized at run time. No calibration data is needed.
- The model runs on CPU with onnxruntime. τ is tuned again on validation for the quantized model.
- Saved to `<storage>/models/student_int8/` (`model.onnx`, tokenizer, `meta.json`). The intermediate fp32 ONNX file is deleted after the comparison.

The export uses the TorchScript exporter (`dynamo=False`). The default dynamo exporter produced a graph that quantization rejects with a `ShapeInferenceError`. onnxruntime's symbolic shape inference is also skipped during the quantization pre-processing, because it cannot resolve this graph.

The quantized model classifies one message at a time, everywhere: threshold tuning, evaluation and the app. Dynamic quantization picks activation scales per batch, padded positions included, so a message's output would depend on the other messages in its batch. On validation, batching 128 messages together changed 21 of 3,120 predictions.

## Results

```bash
python -m src.evaluate
```

`src/evaluate.py` loads every model through `src/models.py`, evaluates it once on the 5,520-message test set with the thresholds tuned on validation, and measures size and latency. It writes [`results.csv`](results.csv).

| Model | Accuracy | OOS recall | In-scope accuracy | Appliance (20) | Size (MB) | Latency (ms) |
|---|---|---|---|---|---|---|
| tf-idf + logistic regression | 0.8707 | 0.890 | 0.8664 | 9/20 | 67.3 | 0.66 |
| Feedforward net on fastText | 0.8250 | 0.859 | 0.8175 | 11/20 | 123.6 | 0.07 |
| BERT-base (teacher) | 0.9217 | 0.893 | 0.9281 | 11/20 | 439.2 | 55.96 |
| DistilBERT student, fp32 | 0.9165 | 0.878 | 0.9250 | 11/20 | 269.0 | 27.16 |
| **DistilBERT student, INT8 ONNX** | 0.9074 | 0.926 | 0.9033 | 13/20 | 68.1 | 2.14 |

- **Hardware:** Apple M1 (4 performance + 4 efficiency cores), 16 GB RAM, macOS 15.7.8, CPU only. PyTorch 2.15 with 4 threads; onnxruntime 1.31.1 with default threads.
- **Latency:** one message per call, 20 untimed warm-up calls, then the mean of 200 timed calls on validation messages. It covers normalized text to class probabilities, including each model's own featurization (tf-idf, word vectors or tokenizer). The shared preprocessing is the same for every model and is excluded. `results.csv` also has the median and 95th percentile.
- **Size:** the files each model needs at inference time. For the FFN, that is the network plus the 100k pretrained word vectors (123 MB of the 124 MB). The Transformer sizes include the tokenizer files.

Where the student's speedup comes from (fp32 ONNX measured separately on the same 200 messages):

| Student variant | Latency (ms) |
|---|---|
| PyTorch, fp32 | 27.2 |
| onnxruntime, fp32 | 5.4–5.8 |
| onnxruntime, INT8 | 2.1 |

About 5× comes from the runtime and about 2.6× from quantization. Using 4 threads instead of the default made no difference.

### Test-set use

- **OOS check (step 6):** before the full evaluation, the 1,000 out-of-scope test messages were scored once, to check the cost of the separate appliance threshold. The result was −0.1 to −0.7 points of OOS recall. The threshold rule was kept, and no threshold was changed because of it.
- **First evaluation:** the INT8 model was first evaluated in batches of 128 messages. That run gave accuracy 0.9111 and OOS recall 0.904.
- **Re-evaluation:** after the batch dependence above was found, the INT8 model's thresholds were retuned on validation at batch size 1, and its test row was measured again. The table shows the batch-size-1 result.
- **Unchanged rows:** the other four rows are unchanged between the two runs.

## App

```bash
python app.py
```

Then open http://127.0.0.1:7860. The app needs the trained models in `<storage>/models/`.

For each message, `src/router.py`:

1. runs the shared preprocessing and shows the normalized text, extracted model codes and spelling corrections;
2. answers with the ELIZA-style rules if the message is small talk;
3. otherwise routes by the quantized student (INT8 ONNX): the predicted intent's domain is the team, and out-of-scope or below-threshold predictions go to a human agent. A predicted small-talk intent gets a canned reply.

The baseline, the feedforward net and the BERT teacher run on the same normalized text and are shown side by side, with each model's intent, confidence, team and latency. Each model applies its own validation-tuned thresholds. The models are warmed up at startup, so the latencies are representative from the first message. They are single measurements, not the benchmark above.

## Tests

```bash
python -m pytest -q
```

## Data sources

- **CLINC150 (`data_oos_plus.json`, `domains.json`):** Larson et al., 2019, *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*, EMNLP-IJCNLP. https://github.com/clinc/oos-eval (CC BY 3.0), commit `828f809`.
- **`data/appliance_examples.csv`:** 100 home-appliance support messages over 5 intents (`appliance_repair`, `appliance_spare_parts`, `appliance_warranty`, `appliance_installation`, `appliance_usage`), 20 each, split 60/20/20 per intent. The first draft was written with an AI assistant; see the AI usage statement.
