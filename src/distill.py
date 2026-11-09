"""Knowledge distillation: distilbert-base-uncased (student) learns from the BERT teacher.

The student is trained on the teacher's softened output distribution
(temperature T) together with the true labels:
    loss = alpha * T^2 * KL(teacher_T || student_T) + (1 - alpha) * weighted CE
The epoch and the out-of-scope threshold are chosen on the validation set.

Usage: python -m src.distill   (after python -m src.train_bert)
"""

import json

from src.paths import MODELS_DIR  # sets HF_HOME before transformers is imported

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import load_split
from src.train_bert import encode, fit_bert, pick_device

TEACHER_DIR = MODELS_DIR / "bert_teacher"
OUTPUT_DIR = MODELS_DIR / "bert_student"
STUDENT_BASE = "distilbert-base-uncased"
ALPHA = 0.5
TEMPERATURE = 2.0


def compute_logits(model, tokenizer, texts, device, batch_size=128):
    model.to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            out.append(model(**encode(tokenizer, texts[i : i + batch_size], device)).logits.float().cpu().numpy())
    return np.concatenate(out)


def main():
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    teacher_meta = json.loads((TEACHER_DIR / "meta.json").read_text())
    classes = np.array(teacher_meta["classes"])
    assert classes.tolist() == sorted(set(y_train))
    device = pick_device()

    teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER_DIR)
    teacher_logits = compute_logits(teacher, AutoTokenizer.from_pretrained(TEACHER_DIR), X_train, device)
    del teacher
    train_acc = (classes[teacher_logits.argmax(1)] == np.array(y_train)).mean()
    print(f"device={device} teacher logits {teacher_logits.shape}, teacher train argmax acc={train_acc:.4f}",
          flush=True)

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_BASE)
    student = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_BASE, num_labels=len(classes),
        id2label=dict(enumerate(classes)), label2id={c: i for i, c in enumerate(classes)},
    )
    result = fit_bert(student, tokenizer, X_train, y_train, X_val, y_val, classes, device=device, log_every=100,
                      teacher_logits=teacher_logits, alpha=ALPHA, temperature=TEMPERATURE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result["model"].save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    meta = {"tau": result["tau"], "appliance_tau": result["appliance_tau"], "epoch": result["epoch"],
            "val_metrics": result["metrics"], "classes": classes.tolist(), "base_model": STUDENT_BASE,
            "teacher": TEACHER_DIR.name,
            "alpha": ALPHA, "temperature": TEMPERATURE}
    (OUTPUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    size = sum(p.stat().st_size for p in OUTPUT_DIR.iterdir()) / 1e6
    m = result["metrics"]
    print(f"saved {OUTPUT_DIR.name}: epoch={result['epoch']} tau={result['tau']:.2f} "
          f"appliance_tau={result['appliance_tau']:.2f} "
          f"in_scope={m['in_scope_accuracy']:.4f} oos_recall={m['oos_recall']:.4f} size={size:.0f} MB")


if __name__ == "__main__":
    main()
