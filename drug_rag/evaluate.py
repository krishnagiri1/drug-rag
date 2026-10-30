"""Evaluation: retrieval quality across configurations, and answer quality with LLM judges."""

from __future__ import annotations

import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from .answer import answer_question, format_context
from .judge import judge_correctness, judge_faithfulness
from .llm import LLMClient
from .retrieve import Retriever

KS = (1, 3, 5, 10)


def _first_rank(hits, pred) -> Optional[int]:
    for h in hits:
        if pred(h.chunk):
            return h.rank
    return None


SALT_WORDS = {
    "hydrochloride", "hcl", "dihydrochloride", "monohydrate", "dihydrate", "trihydrate", "hydrate", "anhydrous",
    "sodium", "potassium", "calcium", "magnesium", "acetate", "sulfate", "tartrate", "bitartrate", "maleate",
    "mesylate", "besylate", "citrate", "phosphate", "succinate", "fumarate", "hydrobromide", "bromide",
    "valerate", "propionate", "dipropionate", "extended-release", "er", "xr",
}


CATIONS = {"sodium", "potassium", "calcium", "magnesium", "lithium", "zinc", "ferrous", "aluminum"}


def drug_key(generic_name: str) -> str:
    """Normalize a generic name so salt/form variants of one drug match ("Duloxetine Hydrochloride" == "Duloxetine").

    Names that start with a cation ("Sodium Chloride", "Magnesium Sulfate") are kept whole, because there the
    salt is the drug itself.
    """
    words = generic_name.lower().replace(",", " ").split()
    if not words or words[0] in CATIONS:  # the salt IS the drug: "Sodium Chloride", "Magnesium Sulfate"
        return " ".join(words)
    core = [w for w in words if w not in SALT_WORDS]
    return " ".join(core or words)


def eval_retrieval(retriever: Retriever, testset: list[dict], mode: str, max_k: int = 10) -> dict:
    """Recall@k and MRR for one (index, mode).

    Main hit = a chunk from the gold section of the same drug (salt/form variants count as the same drug,
    since their labels carry the same information). `strict_rank` additionally requires the exact gold label.
    """
    drug_of = {c.set_id: drug_key(c.generic_name) for c in retriever.chunks}
    per_q = []
    t0 = time.perf_counter()
    for q in testset:
        hits = retriever.search(q["question"], k=max_k, mode=mode)
        gold_drug = drug_of.get(q["gold_set_id"])
        same_drug = lambda c: c.set_id == q["gold_set_id"] or drug_key(c.generic_name) == gold_drug  # noqa: E731
        per_q.append({
            "qid": q["qid"], "kind": q["kind"],
            "rank": _first_rank(hits, lambda c: same_drug(c) and c.section == q["gold_section"]),
            "strict_rank": _first_rank(hits, lambda c: c.set_id == q["gold_set_id"] and c.section == q["gold_section"]),
            "drug_rank": _first_rank(hits, same_drug),
        })
    elapsed = time.perf_counter() - t0
    return {"mode": mode, "chunk_size": retriever.meta["chunk_size"], "n_chunks": retriever.meta["n_chunks"],
            "n": len(per_q), "latency_ms": 1000 * elapsed / max(len(per_q), 1),
            **summarize_ranks(per_q), "by_kind": {k: summarize_ranks([p for p in per_q if p["kind"] == k])
                                                 for k in sorted({p["kind"] for p in per_q})},
            "per_question": per_q}


def summarize_ranks(per_q: list[dict]) -> dict:
    n = len(per_q)
    if not n:
        return {}
    out = {f"recall@{k}": sum(1 for p in per_q if p["rank"] and p["rank"] <= k) / n for k in KS}
    out["mrr@10"] = sum(1 / p["rank"] for p in per_q if p["rank"]) / n
    out["drug_recall@5"] = sum(1 for p in per_q if p["drug_rank"] and p["drug_rank"] <= 5) / n
    if all("strict_rank" in p for p in per_q):
        out["strict_recall@5"] = sum(1 for p in per_q if p["strict_rank"] and p["strict_rank"] <= 5) / n
    return out


def eval_answers(retriever: Retriever, generator: LLMClient, judge: LLMClient, testset: list[dict],
                 mode: str, k: int, out_path: str | Path, workers: int = 1, progress: bool = True) -> list[dict]:
    """Answer each question, then grade faithfulness and correctness. Resumable via `out_path`."""
    out_path = Path(out_path)
    done = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if "error" not in r:
                    done[r["qid"]] = r
    todo = [q for q in testset if q["qid"] not in done]
    lock = threading.Lock()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def work(q: dict) -> dict:
        try:
            a = answer_question(retriever, generator, q["question"], k=k, mode=mode)
            faith = judge_faithfulness(judge, format_context(a.hits), a.answer)
            corr = judge_correctness(judge, q["question"], q["reference_answer"], a.answer)
            gold_drug = next((drug_key(c.generic_name) for c in retriever.chunks if c.set_id == q["gold_set_id"]), None)
            gold_retrieved = any(h.chunk.section == q["gold_section"] and
                                 (h.chunk.set_id == q["gold_set_id"] or drug_key(h.chunk.generic_name) == gold_drug)
                                 for h in a.hits)
            return {"qid": q["qid"], "kind": q["kind"], "question": q["question"], "answer": a.answer,
                    "refused": a.refused, "cited": a.cited, "invalid_citations": a.invalid_citations,
                    "gold_retrieved": gold_retrieved, "faithfulness": faith, "correctness": corr,
                    "retrieval_ms": round(a.retrieval_ms, 1), "generation_ms": round(a.generation_ms, 1),
                    "sources": [s["chunk_id"] for s in a.sources()]}
        except Exception as e:  # keep going; the record is retried on the next run
            return {"qid": q["qid"], "error": f"{type(e).__name__}: {e}"}

    n_done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for f in as_completed([pool.submit(work, q) for q in todo]):
            r = f.result()
            with lock:
                with out_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                if "error" not in r:
                    done[r["qid"]] = r
                n_done += 1
                if progress and n_done % 10 == 0:
                    print(f"  answered {n_done}/{len(todo)}")
    return [done[q["qid"]] for q in testset if q["qid"] in done]


def summarize_answers(records: list[dict]) -> dict:
    n = len(records)
    if not n:
        return {"n": 0}
    answered = [r for r in records if not r["refused"]]
    faith = [r["faithfulness"]["faithfulness"] for r in answered
             if r["faithfulness"].get("faithfulness") is not None]
    claims = [(r["faithfulness"]["n_supported"], r["faithfulness"]["n_claims"]) for r in answered
              if r["faithfulness"].get("n_claims")]
    corr = [r["correctness"]["correctness"] for r in records if "correctness" in r["correctness"]]
    return {
        "n": n,
        "faithfulness_mean": statistics.mean(faith) if faith else None,
        "claims_supported": (sum(s for s, _ in claims) / sum(t for _, t in claims)) if claims else None,
        "fully_faithful_rate": (sum(1 for f in faith if f == 1.0) / len(faith)) if faith else None,
        "correctness_mean": statistics.mean(corr) if corr else None,
        "correct_rate": (sum(1 for c in corr if c == 1.0) / len(corr)) if corr else None,
        "refusal_rate": 1 - len(answered) / n,
        "citation_rate": sum(1 for r in answered if r["cited"]) / len(answered) if answered else None,
        "invalid_citation_rate": sum(1 for r in records if r["invalid_citations"]) / n,
        "gold_retrieved_rate": sum(1 for r in records if r["gold_retrieved"]) / n,
        "median_retrieval_ms": statistics.median(r["retrieval_ms"] for r in records),
        "median_generation_ms": statistics.median(r["generation_ms"] for r in records),
    }
