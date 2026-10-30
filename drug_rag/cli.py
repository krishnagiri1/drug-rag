"""Command line: python -m drug_rag <command> ..."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import (CHUNK_SIZES, DEFAULT_CHUNK_SIZE, DEFAULT_MODE, EMBED_MODEL, INDEX_DIR, LABELS_PATH,
                     RERANK_MODEL, RESULTS_DIR, RETRIEVAL_MODES, TESTSET_PATH)


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-." else "-" for c in s).strip("-")


def _llm(provider, model, base_url=None, json_mode=False):
    from .llm import make_client

    kw = {"json_mode": json_mode} if provider == "ollama" else {}
    return make_client(provider, model, base_url=base_url, **kw)


def _add_llm(p, prefix="", required=True, help_=""):
    p.add_argument(f"--{prefix}provider", required=required, choices=["openai", "anthropic", "ollama", "mock"],
                   help=help_ or "LLM provider")
    p.add_argument(f"--{prefix}model", required=required, help="Model name, e.g. qwen2.5:14b for Ollama")
    p.add_argument(f"--{prefix}base-url", default=None, help="OpenAI-compatible base URL or Ollama host")


def _add_backend(p):
    p.add_argument("--backend", default=os.environ.get("DRUG_RAG_BACKEND", "fastembed"), choices=["fastembed", "hash"],
                   help="fastembed = real ONNX models (default); hash = no-download stand-in for testing")


def cmd_download(a):
    from .fda import download_labels

    download_labels(a.n, max_records=a.max_records, out=a.out)


def cmd_index(a):
    from .embeddings import make_embedder
    from .fda import load_labels
    from .index import build_index

    labels = load_labels(a.labels)
    embedder = make_embedder(a.backend, EMBED_MODEL)
    for cs in a.chunk_sizes:
        build_index(labels, cs, embedder, root=a.index_root)


def cmd_testset(a):
    from .fda import load_labels
    from .testset import build_testset, export_review_csv

    labels = load_labels(a.labels)
    llm = _llm(a.provider, a.model, a.base_url, json_mode=True) if a.provider and a.n_generated else None
    qs = build_testset(labels, llm, a.n_template, a.n_generated if llm else 0, a.out, seed=a.seed)
    kinds = {k: sum(1 for q in qs if q["kind"] == k) for k in ("template", "generated")}
    print(f"Saved {len(qs)} questions {kinds} to {a.out}")
    if kinds["generated"]:
        csv_path = Path(a.out).with_suffix(".review.csv")
        export_review_csv(qs, csv_path)
        print(f"Spot-check generated questions in {csv_path} (set keep=0 on bad ones), then run "
              f"`python -m drug_rag apply-review`.")


def cmd_apply_review(a):
    from .testset import apply_review_csv, load_testset

    qs = load_testset(a.testset)
    kept = apply_review_csv(qs, a.csv or Path(a.testset).with_suffix(".review.csv"))
    with open(a.testset, "w", encoding="utf-8") as f:
        for q in kept:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Kept {len(kept)} of {len(qs)} questions.")


def _retriever(chunk_size, backend, need_reranker, index_root=INDEX_DIR):
    from .embeddings import make_embedder, make_reranker
    from .index import index_dir
    from .retrieve import Retriever

    return Retriever(index_dir(chunk_size, index_root), make_embedder(backend, EMBED_MODEL),
                     make_reranker(backend, RERANK_MODEL) if need_reranker else None)


def cmd_eval_retrieval(a):
    from .evaluate import eval_retrieval
    from .testset import load_testset

    qs = load_testset(a.testset)
    out = Path(a.results)
    out.mkdir(parents=True, exist_ok=True)
    for cs in a.chunk_sizes:
        r = _retriever(cs, a.backend, "hybrid_rerank" in a.modes, a.index_root)
        for mode in a.modes:
            res = eval_retrieval(r, qs, mode)
            (out / f"retrieval_cs{cs}_{mode}.json").write_text(json.dumps(res, indent=1))
            print(f"cs={cs:<5} {mode:<14} recall@1={res['recall@1']:.2f} recall@5={res['recall@5']:.2f} "
                  f"recall@10={res['recall@10']:.2f} mrr={res['mrr@10']:.2f} ({res['latency_ms']:.0f} ms/q)")


def cmd_eval_answers(a):
    from .evaluate import eval_answers, summarize_answers
    from .testset import load_testset

    qs = load_testset(a.testset)
    if a.kind != "all":
        qs = [q for q in qs if q["kind"] == a.kind]
    qs = qs[: a.n]
    r = _retriever(a.chunk_size, a.backend, a.mode == "hybrid_rerank", a.index_root)
    gen = _llm(a.provider, a.model, a.base_url)
    judge = _llm(a.judge_provider or a.provider, a.judge_model or a.model, a.judge_base_url or a.base_url,
                 json_mode=True)
    tag = f"cs{a.chunk_size}_{a.mode}_k{a.k}_{_slug(gen.name)}"
    out = Path(a.results) / f"answers_{tag}.jsonl"
    recs = eval_answers(r, gen, judge, qs, a.mode, a.k, out, workers=a.workers)
    summary = summarize_answers(recs)
    meta = {"chunk_size": a.chunk_size, "mode": a.mode, "k": a.k, "generator": gen.name, "judge": judge.name,
            "summary": summary}
    (Path(a.results) / f"answers_{tag}.summary.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def cmd_report(a):
    from .report import build_report

    print(build_report(a.results, mlflow=not a.no_mlflow))


def cmd_export_site(a):
    from .site import export_site

    page = export_site(a.out, Path(a.results), Path(a.testset), Path(a.index_root))
    print(f"Wrote {page} and {page.parent / 'data.js'}. Open {page} in a browser to preview it.")


def cmd_ask(a):
    from .pipeline import load_pipeline

    p = load_pipeline(a.provider, a.model, a.base_url, a.chunk_size, a.mode, a.k, a.backend)
    ans = p.ask(a.question)
    print(ans.answer, "\n")
    for s in ans.sources():
        print(f"[{s['n']}]{'*' if s['cited'] else ' '} {s['drug']} | {s['section']} | {s['chunk_id']}")


def cmd_serve(a):
    import uvicorn

    uvicorn.run("drug_rag.api:app", host=a.host, port=a.port)


def cmd_demo(a):
    from .pipeline import load_pipeline
    from .ui import build_app

    p = load_pipeline(a.provider, a.model, a.base_url, a.chunk_size, a.mode, a.k, a.backend)
    build_app(p).launch(server_name=a.host, server_port=a.port)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="drug_rag", description="RAG over FDA drug labels")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("download", help="Download prescription drug labels from openFDA")
    p.add_argument("--n", type=int, default=1200, help="Unique drugs (generic names) to keep")
    p.add_argument("--max-records", type=int, default=8000, help="Max raw labels to scan")
    p.add_argument("--out", default=str(LABELS_PATH))
    p.set_defaults(fn=cmd_download)

    p = sub.add_parser("index", help="Chunk the labels and build BM25 + FAISS indexes")
    p.add_argument("--labels", default=str(LABELS_PATH))
    p.add_argument("--chunk-sizes", type=int, nargs="+", default=list(CHUNK_SIZES))
    p.add_argument("--index-root", default=str(INDEX_DIR))
    _add_backend(p)
    p.set_defaults(fn=cmd_index)

    p = sub.add_parser("testset", help="Build the evaluation question set")
    p.add_argument("--labels", default=str(LABELS_PATH))
    p.add_argument("--n-template", type=int, default=50)
    p.add_argument("--n-generated", type=int, default=120, help="Generate extra; you'll drop bad ones in review")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(TESTSET_PATH))
    _add_llm(p, required=False, help_="LLM that writes the generated questions (omit for template-only)")
    p.set_defaults(fn=cmd_testset)

    p = sub.add_parser("apply-review", help="Drop questions marked keep=0 in the review CSV")
    p.add_argument("--testset", default=str(TESTSET_PATH))
    p.add_argument("--csv", default=None)
    p.set_defaults(fn=cmd_apply_review)

    p = sub.add_parser("eval-retrieval", help="Recall@k / MRR for every chunk size x retriever")
    p.add_argument("--testset", default=str(TESTSET_PATH))
    p.add_argument("--chunk-sizes", type=int, nargs="+", default=list(CHUNK_SIZES))
    p.add_argument("--modes", nargs="+", default=list(RETRIEVAL_MODES), choices=list(RETRIEVAL_MODES))
    p.add_argument("--index-root", default=str(INDEX_DIR))
    p.add_argument("--results", default=str(RESULTS_DIR))
    _add_backend(p)
    p.set_defaults(fn=cmd_eval_retrieval)

    p = sub.add_parser("eval-answers", help="Generate answers and grade them with an LLM judge")
    _add_llm(p)
    _add_llm(p, prefix="judge-", required=False, help_="Judge LLM (defaults to the generator)")
    p.add_argument("--testset", default=str(TESTSET_PATH))
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--mode", default=DEFAULT_MODE, choices=list(RETRIEVAL_MODES))
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--kind", default="generated", choices=["generated", "template", "all"])
    p.add_argument("--workers", type=int, default=1, help="Parallel requests (keep 1 for a local Ollama model)")
    p.add_argument("--index-root", default=str(INDEX_DIR))
    p.add_argument("--results", default=str(RESULTS_DIR))
    _add_backend(p)
    p.set_defaults(fn=cmd_eval_answers)

    p = sub.add_parser("report", help="Write results/report.md and log runs to MLflow")
    p.add_argument("--results", default=str(RESULTS_DIR))
    p.add_argument("--no-mlflow", action="store_true")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("export-site", help="Write a static answer explorer (for GitHub Pages) to docs/")
    p.add_argument("--out", default="docs")
    p.add_argument("--results", default=str(RESULTS_DIR))
    p.add_argument("--testset", default=str(TESTSET_PATH))
    p.add_argument("--index-root", default=str(INDEX_DIR))
    p.set_defaults(fn=cmd_export_site)

    for name, fn, port, helptext in (("ask", cmd_ask, None, "Answer one question in the terminal"),
                                     ("demo", cmd_demo, 7860, "Launch the Gradio demo")):
        p = sub.add_parser(name, help=helptext)
        if name == "ask":
            p.add_argument("question")
        _add_llm(p, required=(name == "ask"))
        p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
        p.add_argument("--mode", default=DEFAULT_MODE, choices=list(RETRIEVAL_MODES))
        p.add_argument("--k", type=int, default=5)
        _add_backend(p)
        if port:
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=port)
        p.set_defaults(fn=fn)

    p = sub.add_parser("serve", help="Run the FastAPI server (configure with environment variables)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_serve)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
