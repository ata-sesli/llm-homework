"""Transformer: bert-base-uncased fine-tuned for intent classification (the teacher).

A linear classification head on the [CLS] representation is trained together
with the encoder (AdamW, linear warmup/decay, class-weighted cross-entropy).
The epoch and the out-of-scope threshold are chosen on the validation set.

Usage: python -m src.train_bert
"""

import json
import time

from src.paths import MODELS_DIR  # sets HF_HOME before transformers is imported

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.data import load_split
from src.evaluation import tune_threshold

BASE_MODEL = "bert-base-uncased"
OUTPUT_DIR = MODELS_DIR / "bert_teacher"
MAX_LENGTH = 64
SEED = 42


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def encode(tokenizer, texts, device):
    batch = tokenizer(list(texts), padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    return {k: v.to(device) for k, v in batch.items()}


def predict_proba(model, tokenizer, texts, device, batch_size=128):
    model.to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            logits = model(**encode(tokenizer, texts[i : i + batch_size], device)).logits
            out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(out)


def distillation_loss(student_logits, teacher_logits, labels, ce_loss_fn, alpha, temperature):
    """alpha * T^2 * KL(teacher_T || student_T) + (1 - alpha) * hard-label loss (Hinton et al., 2015)."""
    T = temperature
    soft = F.kl_div(F.log_softmax(student_logits / T, dim=1), F.softmax(teacher_logits / T, dim=1),
                    reduction="batchmean") * T * T
    return alpha * soft + (1 - alpha) * ce_loss_fn(student_logits, labels)


def fit_bert(model, tokenizer, train_texts, train_labels, val_texts, val_labels, classes,
             epochs=5, lr=5e-5, batch_size=32, warmup=0.1, device="cpu", seed=SEED, log_every=0,
             teacher_logits=None, alpha=0.5, temperature=2.0):
    """Fine-tune with class-weighted cross-entropy, or distil when teacher_logits (train order) are given."""
    torch.manual_seed(seed)
    classes = np.asarray(classes)
    index = {c: i for i, c in enumerate(classes)}
    y = torch.tensor([index[c] for c in train_labels])
    counts = torch.bincount(y, minlength=len(classes)).float()
    weights = (len(y) / (len(classes) * counts.clamp(min=1))).to(device)  # balanced, as in the other models

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps_per_epoch = (len(y) + batch_size - 1) // batch_size
    scheduler = get_linear_schedule_with_warmup(optimizer, int(warmup * epochs * steps_per_epoch),
                                                epochs * steps_per_epoch)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    teacher = None if teacher_logits is None else torch.as_tensor(np.asarray(teacher_logits), dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)

    best = None
    for epoch in range(1, epochs + 1):
        model.train()
        start, total = time.perf_counter(), 0.0
        for step, batch in enumerate(torch.randperm(len(y), generator=generator).split(batch_size), 1):
            inputs = encode(tokenizer, [train_texts[i] for i in batch], device)
            logits = model(**inputs).logits
            if teacher is None:
                loss = loss_fn(logits, y[batch].to(device))
            else:
                loss = distillation_loss(logits, teacher[batch].to(device), y[batch].to(device),
                                         loss_fn, alpha, temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(batch)
            if log_every and step % log_every == 0:
                print(f"  epoch {epoch} step {step}/{steps_per_epoch} loss={loss.item():.4f} "
                      f"({time.perf_counter() - start:.0f}s)", flush=True)

        val_probs = predict_proba(model, tokenizer, val_texts, device)
        tau, appliance_tau, metrics = tune_threshold(val_probs, classes, val_labels)
        score = (metrics["in_scope_accuracy"] + metrics["oos_recall"]) / 2
        if best is None or score > best["score"]:
            best = {"score": score, "epoch": epoch, "tau": tau, "appliance_tau": appliance_tau, "metrics": metrics,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        if log_every:
            print(f"epoch {epoch}: loss={total / len(y):.4f} tau={tau:.2f} val_acc={metrics['accuracy']:.4f} "
                  f"in_scope={metrics['in_scope_accuracy']:.4f} oos_recall={metrics['oos_recall']:.4f} "
                  f"score={score:.4f} ({time.perf_counter() - start:.0f}s)", flush=True)

    model.load_state_dict(best["state"])
    return {"model": model, "tau": best["tau"], "appliance_tau": best["appliance_tau"], "metrics": best["metrics"],
            "epoch": best["epoch"], "score": best["score"]}


def main():
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    classes = np.array(sorted(set(y_train)))
    device = pick_device()
    print(f"device={device} train={len(X_train)} val={len(X_val)} classes={len(classes)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(classes),
        id2label=dict(enumerate(classes)), label2id={c: i for i, c in enumerate(classes)},
    )
    result = fit_bert(model, tokenizer, X_train, y_train, X_val, y_val, classes, device=device, log_every=100)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result["model"].save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    meta = {"tau": result["tau"], "appliance_tau": result["appliance_tau"], "epoch": result["epoch"],
            "val_metrics": result["metrics"], "classes": classes.tolist(), "base_model": BASE_MODEL}
    (OUTPUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    size = sum(p.stat().st_size for p in OUTPUT_DIR.iterdir()) / 1e6
    m = result["metrics"]
    print(f"saved {OUTPUT_DIR.name}: epoch={result['epoch']} tau={result['tau']:.2f} "
          f"appliance_tau={result['appliance_tau']:.2f} "
          f"in_scope={m['in_scope_accuracy']:.4f} oos_recall={m['oos_recall']:.4f} size={size:.0f} MB")


if __name__ == "__main__":
    main()
