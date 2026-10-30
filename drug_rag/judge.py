"""LLM-as-judge graders for answer quality.

Faithfulness   Split the answer into claims and check each against the retrieved excerpts.
               score = supported claims / total claims. Measures hallucination.
Correctness    Compare the answer with the test set's reference answer:
               correct (1.0) / partial (0.5) / incorrect (0.0).

Same approach as the agent-judge project: temperature 0, strict JSON replies, a parse
retry with the error message, and every call cached.
"""

from __future__ import annotations

import json
import re

from .llm import LLMClient

SYSTEM = ("You are a meticulous clinical pharmacist who audits AI-generated answers against source documents. "
          "You answer with a single JSON object and nothing else.")

FAITHFULNESS = """JUDGE_FAITHFULNESS
Check whether each factual claim in the ANSWER is supported by the PASSAGES (excerpts from FDA drug labels).
A claim is supported only if the passages state it or directly imply it. Outside medical knowledge does not count,
even if the claim is true. Ignore citation markers like [1] when extracting claims.

PASSAGES:
{context}

ANSWER:
{answer}

Return only JSON: {{"claims": [{{"claim": "<short claim>", "supported": true|false}}], "refusal": true|false}}
Set "refusal" to true if the answer only says the information could not be found."""

CORRECTNESS = """JUDGE_CORRECTNESS
QUESTION: {question}

REFERENCE: {reference}

ANSWER: {answer}

Does the ANSWER convey the key facts of the REFERENCE answer for this QUESTION? Extra correct detail is fine.
Contradicting the reference, or missing its key fact, is not.
Return only JSON: {{"verdict": "correct" | "partial" | "incorrect", "reason": "<one sentence>"}}"""


class ParseError(ValueError):
    pass


def extract_json(text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", text or "")
    a, b = cleaned.find("{"), cleaned.rfind("}")
    if a == -1 or b <= a:
        raise ParseError("no JSON object found")
    try:
        return json.loads(cleaned[a : b + 1])
    except json.JSONDecodeError as e:
        raise ParseError(f"invalid JSON: {e.msg}") from e


def _ask(llm: LLMClient, prompt: str, parse, retries: int = 2):
    p, err = prompt, None
    for _ in range(retries + 1):
        raw = llm.complete(SYSTEM, p)
        try:
            return parse(raw), None
        except (ParseError, KeyError, TypeError, ValueError) as e:
            err = str(e)
            p = prompt + f"\n\nYour previous reply could not be parsed ({err}). Reply with valid JSON only."
    return None, err


def _parse_faith(raw: str) -> dict:
    d = extract_json(raw)
    claims = d["claims"]
    if not isinstance(claims, list):
        raise ParseError("claims must be a list")
    sup = [bool(c.get("supported")) for c in claims if isinstance(c, dict)]
    return {"refusal": bool(d.get("refusal", False)), "n_claims": len(sup), "n_supported": sum(sup),
            "faithfulness": (sum(sup) / len(sup)) if sup else None, "claims": claims}


def _parse_correct(raw: str) -> dict:
    d = extract_json(raw)
    v = str(d["verdict"]).strip().lower()
    if v not in ("correct", "partial", "incorrect"):
        raise ParseError(f"bad verdict {v!r}")
    return {"verdict": v, "correctness": {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}[v],
            "reason": str(d.get("reason", ""))}


def judge_faithfulness(llm: LLMClient, context: str, answer: str) -> dict:
    res, err = _ask(llm, FAITHFULNESS.format(context=context, answer=answer), _parse_faith)
    return res or {"error": err}


def judge_correctness(llm: LLMClient, question: str, reference: str, answer: str) -> dict:
    res, err = _ask(llm, CORRECTNESS.format(question=question, reference=reference, answer=answer), _parse_correct)
    return res or {"error": err}
