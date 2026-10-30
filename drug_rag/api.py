"""FastAPI service.

    uvicorn drug_rag.api:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /health   index and model info
    POST /search   retrieval only (no LLM needed)
    POST /ask      retrieval + cited answer
Interactive docs at /docs.
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import RETRIEVAL_MODES
from .pipeline import Pipeline, load_pipeline

app = FastAPI(title="FDA Drug Label QA", version="0.1.0",
              description="Retrieval-augmented question answering over FDA drug labels. Not medical advice.")
_pipeline: Optional[Pipeline] = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = load_pipeline()
    return _pipeline


class SearchRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    k: int = Field(5, ge=1, le=20)
    mode: Optional[str] = Field(None, description=f"One of {RETRIEVAL_MODES}")


class Source(BaseModel):
    n: int
    chunk_id: str
    drug: str
    brand: str
    section: str
    text: str
    cited: bool = False


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    sources: list[Source]
    retrieval_ms: float
    generation_ms: float
    disclaimer: str = "Generated from FDA label excerpts for information only. Not medical advice."


@app.get("/health")
def health():
    p = get_pipeline()
    return {"status": "ok", "index": p.retriever.meta, "mode": p.mode, "k": p.k,
            "llm": getattr(p.llm, "name", None)}


@app.post("/search", response_model=list[Source])
def search(req: SearchRequest):
    p = get_pipeline()
    mode = req.mode or p.mode
    if mode not in RETRIEVAL_MODES:
        raise HTTPException(422, f"mode must be one of {RETRIEVAL_MODES}")
    if mode == "hybrid_rerank" and p.retriever.reranker is None:
        mode = "hybrid"
    hits = p.retriever.search(req.question, k=req.k, mode=mode)
    return [Source(n=i + 1, chunk_id=h.chunk.chunk_id, drug=h.chunk.generic_name, brand=h.chunk.brand_name,
                   section=h.chunk.section, text=h.chunk.text) for i, h in enumerate(hits)]


@app.post("/ask", response_model=AskResponse)
def ask(req: SearchRequest):
    p = get_pipeline()
    if p.llm is None:
        raise HTTPException(503, "No LLM configured on the server (set LLM_PROVIDER and LLM_MODEL). /search still works.")
    mode = req.mode or p.mode
    if mode not in RETRIEVAL_MODES:
        raise HTTPException(422, f"mode must be one of {RETRIEVAL_MODES}")
    a = p.ask(req.question, k=req.k, mode=mode)
    return AskResponse(question=a.question, answer=a.answer, refused=a.refused,
                       sources=[Source(**s) for s in a.sources()],
                       retrieval_ms=round(a.retrieval_ms, 1), generation_ms=round(a.generation_ms, 1))
