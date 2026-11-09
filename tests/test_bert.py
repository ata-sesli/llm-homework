import numpy as np
import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from src.train_bert import fit_bert, predict_proba

WORDS = ["oven", "broken", "card", "lost", "weather", "today", "my", "is"]
CLASSES = np.array(["appliance_repair", "lost_card", "oos"])


@pytest.fixture
def tiny():
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]
    tokenizer = BertTokenizer(vocab={t: i for i, t in enumerate(tokens)})
    assert tokenizer("oven broken")["input_ids"] == [2, 5, 6, 3]
    config = BertConfig(
        vocab_size=len(WORDS) + 5, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=32, num_labels=len(CLASSES),
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
    )
    torch.manual_seed(0)  # weight init must not depend on what ran earlier in the process
    return BertForSequenceClassification(config), tokenizer


def toy_data():
    texts = ["my oven is broken", "oven broken", "my card is lost", "card lost", "weather today", "today weather"] * 4
    labels = ["appliance_repair", "appliance_repair", "lost_card", "lost_card", "oos", "oos"] * 4
    return texts, labels


def test_predict_proba_shape_and_normalized(tiny):
    model, tokenizer = tiny
    probs = predict_proba(model, tokenizer, ["oven broken", "card", "weather today now"], "cpu", batch_size=2)
    assert probs.shape == (3, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)


def test_fit_bert_learns_toy_task(tiny):
    model, tokenizer = tiny
    texts, labels = toy_data()
    # These settings reached 1.0 from 6 of 6 different weight initialisations.
    result = fit_bert(model, tokenizer, texts, labels, texts, labels, CLASSES,
                      epochs=30, lr=3e-3, batch_size=4, device="cpu", seed=0)
    assert result["metrics"]["accuracy"] == 1.0
    assert 1 <= result["epoch"] <= 30
