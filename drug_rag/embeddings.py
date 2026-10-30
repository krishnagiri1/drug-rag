"""Embedding and reranking backends.

fastembed     ONNX models on CPU (default). Small install, no PyTorch, same code on a laptop,
              in Docker, and on a small cloud instance.
hash          Deterministic bag-of-words hashing "embedder" and term-overlap "reranker".
              No downloads; used by the test suite and for quick dry runs.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import numpy as np

from .config import EMBED_MODEL, RERANK_MODEL

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype("float32")


class FastEmbedder:
    def __init__(self, model: str = EMBED_MODEL, batch_size: int = 64):
        from fastembed import TextEmbedding

        self.name = f"fastembed:{model}"
        self.model = TextEmbedding(model)
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.array(list(self.model.passage_embed(texts, batch_size=self.batch_size))))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(np.array(list(self.model.query_embed([text]))))[0]


class FastReranker:
    def __init__(self, model: str = RERANK_MODEL):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = f"fastembed:{model}"
        self.model = TextCrossEncoder(model)

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [float(s) for s in self.model.rerank(query, passages)]


class HashEmbedder:
    """Feature-hashed bag of words. Crude, but deterministic and dependency-free."""

    def __init__(self, dim: int = 512):
        self.name = f"hash:{dim}"
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) % 2 else -1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._vec(text)[None, :])[0]


class OverlapReranker:
    """Scores passages by query-term overlap. Stand-in for tests."""

    name = "overlap"

    def score(self, query: str, passages: list[str]) -> list[float]:
        q = set(tokenize(query))
        out = []
        for p in passages:
            toks = tokenize(p)
            hits = sum(1 for t in toks if t in q)
            out.append(hits / math.sqrt(len(toks) + 1))
        return out


def make_embedder(backend: str = "fastembed", model: str = EMBED_MODEL) -> Embedder:
    if backend == "fastembed":
        return FastEmbedder(model)
    if backend == "hash":
        return HashEmbedder()
    raise ValueError(f"Unknown embedding backend {backend!r}; use fastembed or hash.")


def make_reranker(backend: str = "fastembed", model: str = RERANK_MODEL) -> Reranker:
    if backend == "fastembed":
        return FastReranker(model)
    if backend == "hash":
        return OverlapReranker()
    raise ValueError(f"Unknown reranker backend {backend!r}; use fastembed or hash.")
