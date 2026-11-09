"""Word vectors for the feedforward network, loaded or trained with gensim.

Two sources are compared on the validation set:
- "pretrained": English fastText wiki-news vectors (gensim-data
  "fasttext-wiki-news-subwords-300"), limited to the 100k most frequent words.
  These are plain word vectors: unknown words are skipped.
- "own": a gensim FastText model trained on our training messages. It builds
  vectors for unknown words from character n-grams.
"""

import numpy as np
from gensim.models import FastText, KeyedVectors

from src.paths import STORAGE  # noqa: F401  sets GENSIM_DATA_DIR before gensim.downloader is imported

PRETRAINED_NAME = "fasttext-wiki-news-subwords-300"
PRETRAINED_LIMIT = 100_000


def average_embeddings(texts, vectors, dim):
    """Mean of the unit-length vectors of each text's tokens; zeros if none is known."""
    X = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        found = []
        for token in text.split():
            try:
                v = np.asarray(vectors[token], dtype=np.float32)
            except KeyError:
                continue
            norm = np.linalg.norm(v)
            if norm > 0:
                found.append(v / norm)
        if found:
            X[i] = np.mean(found, axis=0)
    return X


def load_pretrained_vectors(limit=PRETRAINED_LIMIT):
    import gensim.downloader as api

    path = api.load(PRETRAINED_NAME, return_path=True)
    return KeyedVectors.load_word2vec_format(path, limit=limit)


def train_own_fasttext(texts, dim=100, seed=42):
    sentences = [t.split() for t in texts]
    model = FastText(
        sentences=sentences, vector_size=dim, window=5, min_count=1, min_n=3, max_n=5,
        bucket=100_000, epochs=30, sg=1, seed=seed, workers=1,
    )
    return model.wv
