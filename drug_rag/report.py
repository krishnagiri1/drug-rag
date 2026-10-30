"""Turn saved evaluation results into results/report.md (and optionally MLflow runs)."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .config import RESULTS_DIR

MODE_NAMES = {"bm25": "BM25 (keyword)", "dense": "Dense (embeddings)", "hybrid": "Hybrid (BM25 + dense, RRF)",
              "hybrid_rerank": "Hybrid + cross-encoder rerank"}


def _f(x, digits=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def _p(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100 * x:.0f}%"


def load_results(results_dir: str | Path = RESULTS_DIR) -> tuple[list[dict], list[dict]]:
    d = Path(results_dir)
    retrieval = [json.loads(p.read_text()) for p in sorted(d.glob("retrieval_cs*.json"))]
    answers = [json.loads(p.read_text()) for p in sorted(d.glob("answers_*.summary.json"))]
    return retrieval, answers


def build_report(results_dir: str | Path = RESULTS_DIR, mlflow: bool = False) -> str:
    results_dir = Path(results_dir)
    retrieval, answers = load_results(results_dir)
    L = ["# Results", ""]

    if retrieval:
        n = retrieval[0]["n"]
        kinds = retrieval[0].get("by_kind", {})
        L += ["## Retrieval", "",
              f"{n} test questions ({', '.join(f'{len([1 for p in retrieval[0]['per_question'] if p['kind'] == k])} {k}' for k in kinds)}). "
              "A hit means a retrieved chunk comes from the right drug **and** the right section (salt/form variants such as "
              "\"Duloxetine\" and \"Duloxetine Hydrochloride\" count as the same drug). \"Exact label\" also requires the "
              "specific label the question was written from.", "",
              "| Chunk size | Retriever | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Right drug in top 5 | Recall@5, exact label | Latency (ms/query) |",
              "|---|---|---|---|---|---|---|---|---|"]
        order = {m: i for i, m in enumerate(MODE_NAMES)}
        retrieval.sort(key=lambda r: (r["chunk_size"], order.get(r["mode"], 9)))
        for r in retrieval:
            L.append(f"| {r['chunk_size']} chars | {MODE_NAMES.get(r['mode'], r['mode'])} | {_f(r['recall@1'])} | "
                     f"{_f(r['recall@5'])} | {_f(r['recall@10'])} | {_f(r['mrr@10'])} | {_p(r['drug_recall@5'])} | "
                     f"{_f(r.get('strict_recall@5'))} | {r['latency_ms']:.0f} |")
        best = max(retrieval, key=lambda r: (r["recall@5"], r["mrr@10"]))
        base = min(retrieval, key=lambda r: (r["recall@5"], r["mrr@10"]))
        L += ["", f"Best: **{MODE_NAMES.get(best['mode'])}, {best['chunk_size']}-char chunks**, recall@5 "
                  f"{_f(base['recall@5'])} → {_f(best['recall@5'])} vs. the weakest configuration "
                  f"({MODE_NAMES.get(base['mode'])}, {base['chunk_size']} chars).", ""]
        if len(kinds) > 1:
            L += ["Recall@5 by question type:", "",
                  "| Chunk size | Retriever | " + " | ".join(kinds) + " |", "|---|---|" + "---|" * len(kinds)]
            for r in retrieval:
                L.append(f"| {r['chunk_size']} | {MODE_NAMES.get(r['mode'], r['mode'])} | " +
                         " | ".join(_f(r["by_kind"].get(k, {}).get("recall@5")) for k in kinds) + " |")
            L.append("")

    if answers:
        L += ["## Answer quality (graded by an LLM judge)", "",
              "Faithfulness = share of the answer's claims supported by the retrieved excerpts (hallucination check). "
              "Correctness = agreement with the test set's reference answer (correct 1, partial 0.5, incorrect 0).", "",
              "| Config | Generator | Judge | n | Faithfulness | Fully faithful answers | Correctness | "
              "Refused | Answers with citations | Gold passage retrieved | Median latency (s) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for a in answers:
            s = a["summary"]
            lat = ((s.get("median_retrieval_ms") or 0) + (s.get("median_generation_ms") or 0)) / 1000
            L.append(f"| {a['chunk_size']} chars, {MODE_NAMES.get(a['mode'], a['mode'])}, k={a['k']} | {a['generator']} | "
                     f"{a['judge']} | {s['n']} | {_p(s.get('claims_supported'))} | {_p(s.get('fully_faithful_rate'))} | "
                     f"{_f(s.get('correctness_mean'))} | {_p(s.get('refusal_rate'))} | {_p(s.get('citation_rate'))} | "
                     f"{_p(s.get('gold_retrieved_rate'))} | {lat:.1f} |")
        L.append("")

    if not retrieval and not answers:
        L.append("_No results yet. Run `python -m drug_rag eval-retrieval` first._")

    report = "\n".join(L)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "report.md").write_text(report, encoding="utf-8")
    if mlflow and (retrieval or answers):
        n = log_to_mlflow(retrieval, answers, results_dir)
        report += f"\n\n_Logged {n} runs to MLflow (experiment `drug-rag`); view with `mlflow ui`._"
    return report


def log_to_mlflow(retrieval: list[dict], answers: list[dict], results_dir: Path) -> int:
    import mlflow

    mlflow.set_experiment("drug-rag")
    n = 0
    for r in retrieval:
        with mlflow.start_run(run_name=f"retrieval:cs{r['chunk_size']}:{r['mode']}"):
            mlflow.set_tag("stage", "retrieval")
            mlflow.log_params({"chunk_size": r["chunk_size"], "mode": r["mode"], "n_questions": r["n"],
                               "n_chunks": r["n_chunks"]})
            mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in r.items()
                                if isinstance(v, (int, float)) and k not in ("chunk_size", "n", "n_chunks")})
        n += 1
    for a in answers:
        with mlflow.start_run(run_name=f"answers:cs{a['chunk_size']}:{a['mode']}:k{a['k']}"):
            mlflow.set_tag("stage", "answers")
            mlflow.log_params({k: a[k] for k in ("chunk_size", "mode", "k", "generator", "judge")})
            mlflow.log_metrics({k: v for k, v in a["summary"].items() if isinstance(v, (int, float))})
        n += 1
    with mlflow.start_run(run_name="report"):
        mlflow.set_tag("stage", "report")
        mlflow.log_artifact(str(results_dir / "report.md"))
    return n
