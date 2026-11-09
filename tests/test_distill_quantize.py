import numpy as np
import pytest
import torch
from torch import nn

from src.quantize import OnnxClassifier, export_onnx, quantize_onnx
from src.train_bert import distillation_loss, fit_bert, predict_proba
from tests.test_bert import CLASSES, tiny, toy_data  # noqa: F401  (tiny is a fixture)

STUDENT = torch.tensor([[2.0, 0.5, -1.0], [0.0, 1.0, 0.0]])
TEACHER = torch.tensor([[1.0, 2.0, 0.0], [3.0, 0.0, -1.0]])
LABELS = torch.tensor([1, 0])


def test_distillation_loss_alpha_zero_is_hard_label_loss():
    ce = nn.CrossEntropyLoss()
    assert distillation_loss(STUDENT, TEACHER, LABELS, ce, alpha=0.0, temperature=2.0) == pytest.approx(
        ce(STUDENT, LABELS).item())


def test_distillation_loss_alpha_one_is_scaled_kl():
    T = 2.0
    p_teacher = torch.softmax(TEACHER / T, dim=1)
    kl = (p_teacher * (p_teacher.log() - torch.log_softmax(STUDENT / T, dim=1))).sum(dim=1).mean()
    loss = distillation_loss(STUDENT, TEACHER, LABELS, nn.CrossEntropyLoss(), alpha=1.0, temperature=T)
    assert loss.item() == pytest.approx((T * T * kl).item(), rel=1e-5)


def test_distillation_loss_soft_part_is_zero_when_student_matches_teacher():
    loss = distillation_loss(TEACHER, TEACHER, LABELS, nn.CrossEntropyLoss(), alpha=1.0, temperature=3.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_fit_bert_with_teacher_logits_learns_toy_task(tiny):
    model, tokenizer = tiny
    texts, labels = toy_data()
    index = {c: i for i, c in enumerate(CLASSES)}
    teacher_logits = np.eye(len(CLASSES), dtype=np.float32)[[index[y] for y in labels]] * 5.0
    result = fit_bert(model, tokenizer, texts, labels, texts, labels, CLASSES, epochs=30, lr=3e-3,
                      batch_size=4, device="cpu", seed=0, teacher_logits=teacher_logits)
    assert result["metrics"]["accuracy"] == 1.0


def test_onnx_export_and_int8_quantization_match_pytorch(tiny, tmp_path):
    model, tokenizer = tiny
    texts = ["my oven is broken", "card lost", "weather today", "oven"]
    reference = predict_proba(model, tokenizer, texts, "cpu")

    fp32_path, int8_path = tmp_path / "fp32.onnx", tmp_path / "int8.onnx"
    export_onnx(model, tokenizer, fp32_path)
    quantize_onnx(fp32_path, int8_path)

    np.testing.assert_allclose(OnnxClassifier(fp32_path, tokenizer).predict_proba(texts), reference, atol=1e-5)
    int8_probs = OnnxClassifier(int8_path, tokenizer).predict_proba(texts, batch_size=3)
    np.testing.assert_allclose(int8_probs, reference, atol=0.02)
    assert int8_path.stat().st_size < fp32_path.stat().st_size
