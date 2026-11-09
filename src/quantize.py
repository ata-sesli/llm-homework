"""Export the distilled student to ONNX and quantize it to INT8 (dynamic quantization).

Dynamic quantization stores the weights of the linear (MatMul) layers as 8-bit
integers and quantizes activations on the fly at inference time. No calibration
data is needed. The quantized model runs on CPU with onnxruntime, and its
out-of-scope threshold is tuned again on the validation set.

Usage: python -m src.quantize
"""

import json
import shutil
from pathlib import Path

from src.paths import MODELS_DIR  # sets HF_HOME before transformers is imported

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import load_split
from src.evaluation import tune_threshold
from src.train_bert import MAX_LENGTH

STUDENT_DIR = MODELS_DIR / "bert_student"
OUTPUT_DIR = MODELS_DIR / "student_int8"
INPUT_NAMES = ["input_ids", "attention_mask"]


def export_onnx(model, tokenizer, path):
    """Export with the TorchScript exporter; batch and sequence length stay dynamic."""
    model = model.to("cpu").eval()
    sample = tokenizer(["my oven is broken", "lost card"], padding=True, return_tensors="pt")
    axes = {name: {0: "batch", 1: "sequence"} for name in INPUT_NAMES}
    with torch.no_grad():
        # The dynamo exporter (the default) produced a graph that quantization rejects (ShapeInferenceError).
        torch.onnx.export(model, (sample["input_ids"], sample["attention_mask"]), str(path),
                          input_names=INPUT_NAMES, output_names=["logits"],
                          dynamic_axes={**axes, "logits": {0: "batch"}}, opset_version=17, dynamo=False)


def quantize_onnx(fp32_path, int8_path):
    prepared = Path(int8_path).with_suffix(".prep.onnx")
    # ONNX shape inference and graph optimisation. onnxruntime's symbolic shape inference
    # cannot resolve this graph ("Incomplete symbolic shape inference"), so it is skipped.
    quant_pre_process(str(fp32_path), str(prepared), skip_symbolic_shape=True)
    quantize_dynamic(str(prepared), str(int8_path), weight_type=QuantType.QInt8)
    prepared.unlink()


class OnnxClassifier:
    def __init__(self, model_path, tokenizer, threads=None):
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        self.tokenizer = tokenizer

    def predict_proba(self, texts, batch_size=1):
        # Dynamic quantization picks activation scales per batch (padding included), so a
        # message's output would depend on the other messages in its batch. One message at a
        # time matches the app and makes tuning, evaluation and deployment agree.
        out = []
        for i in range(0, len(texts), batch_size):
            batch = self.tokenizer(list(texts[i : i + batch_size]), padding=True, truncation=True,
                                   max_length=MAX_LENGTH, return_tensors="np")
            logits = self.session.run(None, {name: batch[name].astype(np.int64) for name in INPUT_NAMES})[0]
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            out.append(exp / exp.sum(axis=1, keepdims=True))
        return np.concatenate(out)


def evaluate(name, classifier, X_val, y_val, classes):
    tau, appliance_tau, metrics = tune_threshold(classifier.predict_proba(X_val), classes, y_val)
    print(f"{name}: tau={tau:.2f} appliance_tau={appliance_tau:.2f} "
          f"acc={metrics['accuracy']:.4f} in_scope={metrics['in_scope_accuracy']:.4f} "
          f"oos_recall={metrics['oos_recall']:.4f}", flush=True)
    return tau, appliance_tau, metrics


def main():
    meta = json.loads((STUDENT_DIR / "meta.json").read_text())
    classes = np.array(meta["classes"])
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(STUDENT_DIR)
    X_val, y_val = load_split("val")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fp32_path, int8_path = OUTPUT_DIR / "model_fp32.onnx", OUTPUT_DIR / "model.onnx"
    export_onnx(model, tokenizer, fp32_path)
    quantize_onnx(fp32_path, int8_path)

    evaluate("fp32 onnx", OnnxClassifier(fp32_path, tokenizer), X_val, y_val, classes)
    tau, appliance_tau, metrics = evaluate("int8 onnx", OnnxClassifier(int8_path, tokenizer), X_val, y_val, classes)

    # Only the INT8 model is kept; the fp32 export was needed for quantization and the comparison above.
    fp32_mb, int8_mb = fp32_path.stat().st_size / 1e6, int8_path.stat().st_size / 1e6
    fp32_path.unlink()
    tokenizer.save_pretrained(OUTPUT_DIR)
    out_meta = {"tau": tau, "appliance_tau": appliance_tau, "val_metrics": metrics,
                "classes": classes.tolist(), "source": STUDENT_DIR.name,
                "quantization": "onnxruntime dynamic INT8 (QInt8 weights)"}
    (OUTPUT_DIR / "meta.json").write_text(json.dumps(out_meta, indent=2))
    print(f"saved {OUTPUT_DIR.name}: fp32={fp32_mb:.0f} MB int8={int8_mb:.0f} MB")


if __name__ == "__main__":
    main()
