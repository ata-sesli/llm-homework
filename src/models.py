"""Load every trained model behind one interface, for evaluation and the app.

Each model takes already-normalized texts (see src/preprocess.py) and does its own
featurization: tf-idf, averaged word vectors, or a subword tokenizer.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.paths import MODELS_DIR  # sets HF_HOME before transformers is imported

import numpy as np

from src.evaluation import predict_with_threshold


@dataclass
class RouterModel:
    name: str
    classes: np.ndarray
    tau: float
    appliance_tau: float
    predict_proba: Callable  # list of texts -> (n, n_classes) probabilities
    files: list = field(default_factory=list)  # what is counted as the model's size

    def predict(self, texts):
        return predict_with_threshold(self.predict_proba(texts), self.classes, self.tau, self.appliance_tau)


def size_mb(paths):
    total = 0
    for path in map(Path, paths):
        files = path.rglob("*") if path.is_dir() else [path]
        total += sum(f.stat().st_size for f in files if f.is_file())
    return total / 1e6


def load_baseline():
    import joblib
    saved = joblib.load(MODELS_DIR / "baseline.joblib")
    pipeline = saved["model"]
    return RouterModel("tf-idf + logistic regression", np.asarray(pipeline.classes_), saved["tau"],
                       saved["appliance_tau"], pipeline.predict_proba, [MODELS_DIR / "baseline.joblib"])


def load_ffn():
    import torch
    from gensim.models import KeyedVectors
    from src.embeddings import average_embeddings
    from src.train_ffn import IntentFFN, predict_proba

    saved = torch.load(MODELS_DIR / "ffn.pt")
    model = IntentFFN(saved["input_dim"], saved["hidden_dim"], len(saved["classes"]), saved["dropout"])
    model.load_state_dict(saved["state_dict"])
    vectors = KeyedVectors.load(str(MODELS_DIR / "ffn_vectors.kv"))
    files = [MODELS_DIR / "ffn.pt", *MODELS_DIR.glob("ffn_vectors.kv*")]
    return RouterModel("feedforward net on fastText", np.array(saved["classes"]), saved["tau"], saved["appliance_tau"],
                       lambda texts: predict_proba(model, average_embeddings(texts, vectors, saved["input_dim"])),
                       files)


def _meta(directory):
    return json.loads((directory / "meta.json").read_text())


def load_transformer(name, directory):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from src.train_bert import predict_proba

    meta = _meta(directory)
    model = AutoModelForSequenceClassification.from_pretrained(directory).eval()
    tokenizer = AutoTokenizer.from_pretrained(directory)
    return RouterModel(name, np.array(meta["classes"]), meta["tau"], meta["appliance_tau"],
                       lambda texts: predict_proba(model, tokenizer, texts, "cpu"), [directory])


def load_quantized():
    from transformers import AutoTokenizer
    from src.quantize import OnnxClassifier

    directory = MODELS_DIR / "student_int8"
    meta = _meta(directory)
    classifier = OnnxClassifier(directory / "model.onnx", AutoTokenizer.from_pretrained(directory))
    return RouterModel("DistilBERT student, INT8 ONNX", np.array(meta["classes"]), meta["tau"],
                       meta["appliance_tau"], classifier.predict_proba, [directory])


LOADERS = {
    "baseline": load_baseline,
    "ffn": load_ffn,
    "bert": lambda: load_transformer("BERT-base (teacher)", MODELS_DIR / "bert_teacher"),
    "student": lambda: load_transformer("DistilBERT student, fp32", MODELS_DIR / "bert_student"),
    "student_int8": load_quantized,
}


def load_models(keys=tuple(LOADERS)):
    return {key: LOADERS[key]() for key in keys}
