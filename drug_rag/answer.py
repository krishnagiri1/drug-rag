"""Grounded answer generation with numbered citations."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .llm import LLMClient
from .retrieve import Hit, Retriever

NOT_FOUND = "I could not find this in the provided drug labels."

SYSTEM = (
    "You answer questions about prescription drugs using only excerpts from official FDA drug labels. "
    "You never use outside knowledge, and you never give personal medical advice."
)

PROMPT = """Answer the question using only the numbered label excerpts below.

Rules:
- Every sentence must end with the citation(s) of the excerpt(s) that support it, like [1] or [2][3].
- If the excerpts do not contain the answer, reply exactly: "{not_found}"
- Be concise: at most 5 sentences. Quote numbers (doses, percentages) exactly as written.

EXCERPTS:
{context}

QUESTION: {question}

ANSWER:"""


@dataclass
class Answer:
    question: str
    answer: str
    hits: list[Hit]
    cited: list[int] = field(default_factory=list)  # 1-based passage numbers cited in the answer
    invalid_citations: list[int] = field(default_factory=list)
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0

    @property
    def refused(self) -> bool:
        return NOT_FOUND.lower().rstrip(".") in self.answer.lower()

    def sources(self) -> list[dict]:
        return [{"n": i + 1, "chunk_id": h.chunk.chunk_id, "drug": h.chunk.generic_name,
                 "brand": h.chunk.brand_name, "section": h.chunk.section, "text": h.chunk.text,
                 "cited": (i + 1) in self.cited} for i, h in enumerate(self.hits)]


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i + 1}] {h.chunk.text}" for i, h in enumerate(hits))


def parse_citations(text: str, n_passages: int) -> tuple[list[int], list[int]]:
    nums = [int(x) for x in re.findall(r"\[(\d+)\]", text)]
    valid = sorted({n for n in nums if 1 <= n <= n_passages})
    invalid = sorted({n for n in nums if not 1 <= n <= n_passages})
    return valid, invalid


def build_prompt(question: str, hits: list[Hit]) -> str:
    return PROMPT.format(not_found=NOT_FOUND, context=format_context(hits), question=question.strip())


def answer_question(retriever: Retriever, llm: LLMClient, question: str, k: int = 5,
                    mode: str = "hybrid_rerank") -> Answer:
    t0 = time.perf_counter()
    hits = retriever.search(question, k=k, mode=mode)
    t1 = time.perf_counter()
    text = llm.complete(SYSTEM, build_prompt(question, hits)).strip()
    t2 = time.perf_counter()
    cited, invalid = parse_citations(text, len(hits))
    return Answer(question, text, hits, cited, invalid, (t1 - t0) * 1000, (t2 - t1) * 1000)
