"""End-to-end with the hash backend and the mock LLM: no downloads, no network, no API key."""

import json

from fastapi.testclient import TestClient

from drug_rag.cli import main
from drug_rag.evaluate import eval_answers, eval_retrieval, summarize_answers
from drug_rag.embeddings import make_embedder, make_reranker
from drug_rag.index import index_dir
from drug_rag.llm import make_client
from drug_rag.report import build_report
from drug_rag.retrieve import Retriever
from drug_rag.testset import apply_review_csv, build_testset, export_review_csv, load_testset


def test_template_testset_has_gold(built):
    _, qs = built
    assert len(qs) == 20
    for q in qs:
        assert q["gold_set_id"].startswith("set-") and q["gold_section"]
        assert q["kind"] == "template"


def test_generated_testset_and_review(tmp_path, labels):
    llm = make_client("mock", "x", cache=False)
    qs = build_testset(labels, llm, 4, 6, tmp_path / "t.jsonl")
    gen = [q for q in qs if q["kind"] == "generated"]
    assert len(gen) == 6 and all(q["source_chunk_id"].startswith(q["gold_set_id"]) for q in gen)
    export_review_csv(qs, tmp_path / "r.csv")
    text = (tmp_path / "r.csv").read_text().replace(f"{gen[0]['qid']},1,", f"{gen[0]['qid']},0,")
    (tmp_path / "r.csv").write_text(text)
    assert len(apply_review_csv(qs, tmp_path / "r.csv")) == len(qs) - 1


def test_retrieval_eval_metrics(built):
    root, qs = built
    r = Retriever(index_dir(300, root / "data" / "index"), make_embedder("hash"), make_reranker("hash"))
    res = eval_retrieval(r, qs, "hybrid_rerank")
    assert res["n"] == 20 and 0 <= res["recall@1"] <= res["recall@5"] <= res["recall@10"] <= 1
    assert res["drug_recall@5"] >= res["recall@5"]
    assert res["recall@5"] > 0.5  # four distinctive drugs: even hashing should find most


def test_answer_eval_and_summary(built):
    root, qs = built
    r = Retriever(index_dir(600, root / "data" / "index"), make_embedder("hash"), make_reranker("hash"))
    llm = make_client("mock", "x", cache=False)
    out = root / "results" / "answers.jsonl"
    recs = eval_answers(r, llm, llm, qs[:8], "hybrid_rerank", 3, out, progress=False)
    assert len(recs) == 8 and all(rec["cited"] == [1] for rec in recs)
    s = summarize_answers(recs)
    assert s["n"] == 8 and s["citation_rate"] == 1.0 and s["faithfulness_mean"] is not None
    assert len(eval_answers(r, llm, llm, qs[:8], "hybrid_rerank", 3, out, progress=False)) == 8  # resume
    assert len(out.read_text().strip().splitlines()) == 8


def test_cli_end_to_end_and_report(built, monkeypatch, capsys):
    root, _ = built
    monkeypatch.chdir(root)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    main(["eval-retrieval", "--backend", "hash", "--chunk-sizes", "300", "600"])
    main(["eval-answers", "--backend", "hash", "--provider", "mock", "--model", "x", "--chunk-size", "600",
          "--kind", "template", "--n", "6", "--k", "3"])
    main(["report", "--no-mlflow"])
    report = (root / "results" / "report.md").read_text()
    assert "Retrieval" in report and "Answer quality" in report and "Hybrid + cross-encoder rerank" in report
    assert len(list((root / "results").glob("retrieval_cs*.json"))) == 8


def test_api(built, monkeypatch):
    root, _ = built
    monkeypatch.chdir(root)
    monkeypatch.setenv("DRUG_RAG_BACKEND", "hash")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("LLM_MODEL", "x")
    import drug_rag.api as api
    from drug_rag.pipeline import load_pipeline

    monkeypatch.setattr(api, "_pipeline", load_pipeline(chunk_size=600, index_root=root / "data" / "index"))
    c = TestClient(api.app)
    assert c.get("/health").json()["status"] == "ok"
    hits = c.post("/search", json={"question": "warfarin pregnancy", "k": 3}).json()
    assert len(hits) == 3 and hits[0]["drug"] == "Warfarin Sodium"
    r = c.post("/ask", json={"question": "What are the contraindications for amoxicillin?"}).json()
    assert r["answer"].endswith("[1]") and r["sources"][0]["cited"] is True
    assert c.post("/ask", json={"question": "x"}).status_code == 422


def test_export_site(built, monkeypatch):
    root, _ = built
    monkeypatch.chdir(root)
    main(["eval-answers", "--backend", "hash", "--provider", "mock", "--model", "x", "--chunk-size", "600",
          "--kind", "template", "--n", "5", "--k", "3"])
    main(["report", "--no-mlflow"])
    main(["export-site"])
    js = (root / "docs" / "data.js").read_text(encoding="utf-8")
    data = json.loads(js[len("window.DRUG_RAG_DATA = "):].rstrip().rstrip(";"))
    assert len(data["items"]) == 5 and data["configs"]
    first = next(iter(data["items"][0]["answers"].values()))
    assert first["sources"] and first["sources"][0]["text"]
    assert "<table>" in data["report"]
