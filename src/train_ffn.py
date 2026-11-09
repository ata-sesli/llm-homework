"""Neural network: feedforward classifier on averaged fastText word vectors.

Input: mean of the unit-length word vectors of a message.
Network: Linear -> ReLU -> Dropout -> Linear -> softmax, trained with
mini-batch SGD (momentum 0.9) and backpropagation on a class-weighted
cross-entropy loss. The epoch, the word-vector source and the out-of-scope
threshold are chosen on the validation set.

Usage: python -m src.train_ffn
"""

import copy
import time

import numpy as np
import torch
from torch import nn

from src.data import load_split
from src.embeddings import average_embeddings, load_pretrained_vectors, train_own_fasttext
from src.evaluation import tune_threshold
from src.paths import MODELS_DIR

SEED = 42


class IntentFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(X)), dim=1).numpy()


def fit_ffn(X_train, y_train, X_val, y_val, classes, hidden_dim=512, dropout=0.3,
            epochs=150, lr=0.1, batch_size=64, seed=SEED, log_every=0):
    torch.manual_seed(seed)
    classes = np.asarray(classes)
    index = {c: i for i, c in enumerate(classes)}
    y_idx = torch.tensor([index[c] for c in y_train])
    X_t = torch.from_numpy(X_train)

    counts = torch.bincount(y_idx, minlength=len(classes)).float()
    weights = len(y_idx) / (len(classes) * counts.clamp(min=1))  # balanced, as in the baseline

    model = IntentFFN(X_train.shape[1], hidden_dim, len(classes), dropout)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    generator = torch.Generator().manual_seed(seed)

    best = None
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in torch.randperm(len(y_idx), generator=generator).split(batch_size):
            optimizer.zero_grad()
            loss = loss_fn(model(X_t[batch]), y_idx[batch])
            loss.backward()  # backpropagation
            optimizer.step()  # SGD update
            total += loss.item() * len(batch)

        tau, metrics = tune_threshold(predict_proba(model, X_val), classes, y_val)
        score = (metrics["in_scope_accuracy"] + metrics["oos_recall"]) / 2
        if best is None or score > best["score"]:
            best = {"score": score, "epoch": epoch, "tau": tau, "metrics": metrics,
                    "state": copy.deepcopy(model.state_dict())}
        if log_every and epoch % log_every == 0:
            print(f"  epoch {epoch:3d} loss={total / len(y_idx):.4f} val_score={score:.4f} "
                  f"(best {best['score']:.4f} @ {best['epoch']})", flush=True)

    model.load_state_dict(best["state"])
    return {"model": model, "tau": best["tau"], "metrics": best["metrics"],
            "epoch": best["epoch"], "score": best["score"], "optimizer": optimizer}


def main():
    X_train_text, y_train = load_split("train")
    X_val_text, y_val = load_split("val")
    classes = np.array(sorted(set(y_train)))

    sources = {
        "own": lambda: train_own_fasttext(X_train_text, seed=SEED),
        "pretrained": load_pretrained_vectors,
    }
    results = {}
    for name, load in sources.items():
        start = time.perf_counter()
        vectors = load()
        dim = vectors.vector_size
        X_train = average_embeddings(X_train_text, vectors, dim)
        X_val = average_embeddings(X_val_text, vectors, dim)
        print(f"[{name}] dim={dim} vectors ready ({time.perf_counter() - start:.0f}s)", flush=True)
        result = fit_ffn(X_train, y_train, X_val, y_val, classes, log_every=25)
        m = result["metrics"]
        print(f"[{name}] best epoch={result['epoch']} tau={result['tau']:.2f} val_acc={m['accuracy']:.4f} "
              f"in_scope={m['in_scope_accuracy']:.4f} oos_recall={m['oos_recall']:.4f} "
              f"({time.perf_counter() - start:.0f}s)", flush=True)
        results[name] = (result, vectors)

    name = max(results, key=lambda k: results[k][0]["score"])
    result, vectors = results[name]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    vectors_path = MODELS_DIR / "ffn_vectors.kv"
    vectors.save(str(vectors_path))
    model = result["model"]
    torch.save({
        "state_dict": model.state_dict(),
        "classes": classes.tolist(),
        "tau": result["tau"],
        "input_dim": vectors.vector_size,
        "hidden_dim": model.net[0].out_features,
        "dropout": model.net[2].p,
        "vectors": name,
    }, MODELS_DIR / "ffn.pt")
    size = sum(p.stat().st_size for p in MODELS_DIR.glob("ffn*")) / 1e6
    print(f"saved ffn.pt + {vectors_path.name} (vectors={name}): {size:.1f} MB total")


if __name__ == "__main__":
    main()
