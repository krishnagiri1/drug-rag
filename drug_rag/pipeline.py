"""Load everything needed to answer questions (used by the API, the demo and the CLI)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .answer import Answer, answer_question
from .config import DEFAULT_CHUNK_SIZE, DEFAULT_MODE, EMBED_MODEL, INDEX_DIR, RERANK_MODEL
from .embeddings import make_embedder, make_reranker
from .index import index_dir
from .llm import LLMClient, make_client
from .retrieve import Retriever


@dataclass
class Pipeline:
    retriever: Retriever
    llm: Optional[LLMClient]
    mode: str
    k: int

    def ask(self, question: str, k: Optional[int] = None, mode: Optional[str] = None) -> Answer:
        if self.llm is None:
            raise RuntimeError("No LLM configured; set LLM_PROVIDER and LLM_MODEL (or use --provider/--model).")
        return answer_question(self.retriever, self.llm, question, k=k or self.k, mode=mode or self.mode)


def load_pipeline(provider: Optional[str] = None, model: Optional[str] = None, base_url: Optional[str] = None,
                  chunk_size: int = DEFAULT_CHUNK_SIZE, mode: str = DEFAULT_MODE, k: int = 5,
                  backend: Optional[str] = None, index_root=INDEX_DIR) -> Pipeline:
    """Settings fall back to environment variables, which is how Docker and Spaces configure it."""
    backend = backend or os.environ.get("DRUG_RAG_BACKEND", "fastembed")
    provider = provider or os.environ.get("LLM_PROVIDER")
    model = model or os.environ.get("LLM_MODEL")
    base_url = base_url or os.environ.get("LLM_BASE_URL")
    embedder = make_embedder(backend, EMBED_MODEL)
    reranker = make_reranker(backend, RERANK_MODEL) if mode == "hybrid_rerank" else None
    retriever = Retriever(index_dir(chunk_size, index_root), embedder, reranker)
    llm = make_client(provider, model, base_url=base_url) if provider and model else None
    return Pipeline(retriever, llm, mode, k)
