"""Retrieval: BM25, dense, hybrid (reciprocal rank fusion), and hybrid + cross-encoder rerank.

Why each piece exists
- BM25 (keyword) is strong on exact drug names and rare terms ("QT prolongation").
- Dense embeddings handle paraphrase ("can I take this while pregnant?" vs "Pregnancy").
- Reciprocal rank fusion (RRF) merges the two ranked lists without having to calibrate
  their very different score scales: score(d) = sum over lists of 1 / (k + rank(d)).
- A cross-encoder reranker reads the query and passage together, which is more accurate
  than comparing two independent embeddings but too slow to run on every chunk, so it
  only re-scores the top fused candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .chunking import Chunk
from .config import CANDIDATES, RETRIEVAL_MODES, RRF_K
from .embeddings import Embedder, Reranker, tokenize
from .index import load_chunks, load_meta


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for r, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + r + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


class Retriever:
    def __init__(self, index_path: str | Path, embedder: Embedder, reranker: Optional[Reranker] = None):
        import faiss
        from rank_bm25 import BM25Okapi

        self.path = Path(index_path)
        self.meta = load_meta(self.path)
        if self.meta["embedder"] != embedder.name:
            raise ValueError(f"Index at {self.path} was built with {self.meta['embedder']}, "
                             f"but the query embedder is {embedder.name}. Rebuild the index or switch embedder.")
        self.chunks = load_chunks(self.path)
        self.faiss = faiss.read_index(str(self.path / "dense.faiss"))
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])
        self.embedder = embedder
        self.reranker = reranker

    # --- individual retrievers --------------------------------------------
    def bm25_ranking(self, query: str, n: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        n = min(n, len(scores))
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]

    def dense_ranking(self, query: str, n: int) -> list[tuple[int, float]]:
        q = self.embedder.embed_query(query)[None, :].astype("float32")
        scores, idx = self.faiss.search(q, min(n, len(self.chunks)))
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]

    # --- public API ---------------------------------------------------------
    def search(self, query: str, k: int = 5, mode: str = "hybrid_rerank", candidates: int = CANDIDATES) -> list[Hit]:
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"mode must be one of {RETRIEVAL_MODES}")
        if mode == "bm25":
            ranked = self.bm25_ranking(query, k)
        elif mode == "dense":
            ranked = self.dense_ranking(query, k)
        else:
            n = max(candidates, k)
            fused = rrf([[i for i, _ in self.bm25_ranking(query, n)], [i for i, _ in self.dense_ranking(query, n)]])
            ranked = fused[:n] if mode == "hybrid_rerank" else fused[:k]
            if mode == "hybrid_rerank":
                if self.reranker is None:
                    raise ValueError("hybrid_rerank needs a reranker")
                ids = [i for i, _ in ranked]
                scores = self.reranker.score(query, [self.chunks[i].text for i in ids])
                ranked = sorted(zip(ids, scores), key=lambda x: -x[1])[:k]
        return [Hit(self.chunks[i], s, r + 1) for r, (i, s) in enumerate(ranked)]
