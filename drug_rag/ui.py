"""Gradio demo: ask a question, see the cited answer and the label excerpts behind it."""

from __future__ import annotations

from pathlib import Path

from .config import RESULTS_DIR, SECTIONS
from .pipeline import Pipeline

EXAMPLES = [
    "What are the contraindications for metformin?",
    "Can warfarin be used during pregnancy?",
    "What is the maximum daily dose of acetaminophen?",
    "Which drugs interact with simvastatin?",
]


def _sources_md(sources: list[dict]) -> str:
    out = []
    for s in sources:
        mark = "**cited** · " if s["cited"] else ""
        body = s["text"].split("\n", 1)[-1]
        out.append(f"**[{s['n']}] {s['drug']}{' (' + s['brand'] + ')' if s['brand'] else ''} — "
                   f"{SECTIONS.get(s['section'], s['section'])}** · {mark}`{s['chunk_id']}`\n\n> {body[:900]}")
    return "\n\n".join(out)


def build_app(pipeline: Pipeline):
    import gradio as gr

    def run(question: str, k: int, mode: str):
        if not question.strip():
            return "Type a question first.", ""
        if pipeline.llm is None:
            hits = pipeline.retriever.search(question, k=int(k), mode=mode)
            srcs = [{"n": i + 1, "chunk_id": h.chunk.chunk_id, "drug": h.chunk.generic_name,
                     "brand": h.chunk.brand_name, "section": h.chunk.section, "text": h.chunk.text, "cited": False}
                    for i, h in enumerate(hits)]
            return "_No LLM configured: showing retrieved excerpts only._", _sources_md(srcs)
        a = pipeline.ask(question, k=int(k), mode=mode)
        timing = f"\n\n<sub>retrieval {a.retrieval_ms:.0f} ms · generation {a.generation_ms / 1000:.1f} s</sub>"
        return a.answer + timing, _sources_md(a.sources())

    report = Path(RESULTS_DIR) / "report.md"
    modes = ["bm25", "dense", "hybrid"] + (["hybrid_rerank"] if pipeline.retriever.reranker else [])
    with gr.Blocks(title="FDA Drug Label QA") as app:
        gr.Markdown("# FDA Drug Label QA\nAnswers come only from official FDA label excerpts, with citations. "
                    f"Index: {pipeline.retriever.meta['n_labels']} labels, {pipeline.retriever.meta['n_chunks']} chunks. "
                    "**For information only, not medical advice.**")
        with gr.Tab("Ask"):
            with gr.Row():
                q = gr.Textbox(label="Question", placeholder=EXAMPLES[0], scale=4)
                go = gr.Button("Ask", variant="primary", scale=1)
            with gr.Row():
                k = gr.Slider(1, 10, value=pipeline.k, step=1, label="Excerpts to retrieve (k)")
                mode = gr.Dropdown(modes, value=pipeline.mode if pipeline.mode in modes else "hybrid", label="Retriever")
            gr.Examples(EXAMPLES, inputs=q)
            ans = gr.Markdown()
            srcs = gr.Markdown()
            go.click(run, [q, k, mode], [ans, srcs])
            q.submit(run, [q, k, mode], [ans, srcs])
        with gr.Tab("Evaluation results"):
            gr.Markdown(report.read_text(encoding="utf-8") if report.exists() else "Run the evaluation to see results.")
    return app
