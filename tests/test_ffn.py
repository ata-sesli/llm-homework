import numpy as np
import torch

from src.embeddings import average_embeddings
from src.train_ffn import IntentFFN, fit_ffn


class FakeVectors(dict):
    """Behaves like gensim KeyedVectors: KeyError for unknown words."""


VECS = FakeVectors(oven=np.array([1.0, 0.0]), broken=np.array([0.0, 2.0]))


def test_average_embeddings_skips_unknown_words():
    # Vectors are scaled to unit length before averaging: broken [0, 2] -> [0, 1].
    X = average_embeddings(["oven broken", "oven xyz", "xyz", ""], VECS, dim=2)
    assert X.dtype == np.float32
    np.testing.assert_allclose(X, [[0.5, 0.5], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])


def test_intent_ffn_shape_and_dropout():
    model = IntentFFN(input_dim=4, hidden_dim=8, n_classes=3, dropout=0.5)
    assert model(torch.zeros(5, 4)).shape == (5, 3)
    assert any(isinstance(m, torch.nn.Dropout) for m in model.modules())


def test_fit_ffn_learns_a_separable_problem():
    rng = np.random.default_rng(0)
    centers = np.array([[3, 0], [0, 3], [-3, -3]], dtype=np.float32)
    labels = np.repeat(["a", "b", "oos"], 30)
    X = (centers[np.repeat(np.arange(3), 30)] + rng.normal(0, 0.3, (90, 2))).astype(np.float32)
    classes = np.array(["a", "b", "oos"])
    result = fit_ffn(X, labels, X, labels, classes, hidden_dim=16, epochs=30, lr=0.1, seed=0)
    assert result["metrics"]["accuracy"] == 1.0
    assert isinstance(result["optimizer"], torch.optim.SGD)
