"""Build the evaluation question set.

Two kinds of questions, both with a known gold (drug label, section):

template   "What are the contraindications for warfarin?" -> (warfarin, contraindications).
           Deterministic, no LLM needed. Tests that retrieval finds the right drug and section.
generated  An LLM reads one random passage and writes a specific, natural question it answers,
           plus a short reference answer, e.g. "What should be monitored when starting
           warfarin in elderly patients?". Tests paraphrase and detail-level retrieval.

Gold is defined at the (label, section) level rather than the exact chunk, so the same test
set fairly scores indexes built with different chunk sizes.

Generated questions should be spot-checked by a human before you rely on them; `review`
exports them to a CSV for that.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from .chunking import chunk_labels, drug_title
from .config import SECTIONS
from .judge import ParseError, extract_json
from .llm import LLMClient

TEMPLATES = {
    "indications_and_usage": "What is {drug} indicated for?",
    "contraindications": "What are the contraindications for {drug}?",
    "boxed_warning": "Does {drug} have a boxed warning, and what does it say?",
    "dosage_and_administration": "What is the recommended dosage of {drug}?",
    "adverse_reactions": "What are the most common adverse reactions to {drug}?",
    "drug_interactions": "What drug interactions are listed for {drug}?",
    "overdosage": "What does the {drug} label say about overdosage?",
    "mechanism_of_action": "How does {drug} work?",
}

GEN_SYSTEM = "You write evaluation questions for a drug-information search system. You reply with JSON only."

GEN_PROMPT = """WRITE_QUESTION
Read this excerpt from an FDA drug label and write ONE question that a pharmacist or patient might ask,
which this excerpt answers.

Requirements:
- Name the drug in the question (generic or brand name).
- Ask about a specific fact (a dose, a condition, a population, a risk), not "what does the label say".
- Do not copy a full sentence from the excerpt; paraphrase naturally.
- The reference answer must be 1-2 sentences, fully supported by the excerpt.

PASSAGE:
{passage}

Return only JSON: {{"question": "...", "answer": "..."}}"""


def template_questions(labels: list[dict], n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    pool = [(lab, sec) for lab in labels for sec in TEMPLATES if sec in lab["sections"]]
    rng.shuffle(pool)
    out = []
    for lab, sec in pool[:n]:
        name = rng.choice([lab["generic_name"], lab.get("brand_name") or lab["generic_name"]])
        out.append({
            "qid": f"t{len(out):04d}", "kind": "template",
            "question": TEMPLATES[sec].format(drug=name.lower() if rng.random() < 0.5 else name),
            "reference_answer": lab["sections"][sec][:500],
            "gold_set_id": lab["set_id"], "gold_section": sec, "drug": drug_title(lab),
        })
    return out


def generated_questions(labels: list[dict], llm: LLMClient, n: int, seed: int = 0,
                        chunk_size: int = 1000, progress: bool = True) -> list[dict]:
    rng = random.Random(seed)
    by_set = {lab["set_id"]: lab for lab in labels}
    chunks = [c for c in chunk_labels(labels, chunk_size) if len(c.text) > 300]
    rng.shuffle(chunks)
    out: list[dict] = []
    for c in chunks:
        if len(out) >= n:
            break
        raw = llm.complete(GEN_SYSTEM, GEN_PROMPT.format(passage=c.text))
        try:
            d = extract_json(raw)
            q, a = str(d["question"]).strip(), str(d["answer"]).strip()
        except (ParseError, KeyError):
            continue
        lab = by_set[c.set_id]
        names = {lab["generic_name"].lower(), (lab.get("brand_name") or "").lower()} - {""}
        if len(q) < 15 or not any(nm.split()[0] in q.lower() for nm in names):
            continue  # must name the drug, otherwise the question is ambiguous across 1,200 labels
        out.append({
            "qid": f"g{len(out):04d}", "kind": "generated", "question": q, "reference_answer": a,
            "gold_set_id": c.set_id, "gold_section": c.section, "drug": drug_title(lab),
            "source_chunk_id": c.chunk_id, "source_text": c.text,
        })
        if progress and len(out) % 10 == 0:
            print(f"  generated {len(out)}/{n}")
    return out


def build_testset(labels: list[dict], llm: LLMClient | None, n_template: int, n_generated: int,
                  out: str | Path, seed: int = 0) -> list[dict]:
    qs = template_questions(labels, n_template, seed)
    if llm is not None and n_generated:
        qs += generated_questions(labels, llm, n_generated, seed)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for q in qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    return qs


def load_testset(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def export_review_csv(testset: list[dict], path: str | Path) -> None:
    """CSV for spot-checking generated questions; set keep=0 on bad rows, then run `apply-review`."""
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "keep", "question", "reference_answer", "drug", "section", "source_text"])
        for q in testset:
            if q["kind"] == "generated":
                w.writerow([q["qid"], 1, q["question"], q["reference_answer"], q["drug"],
                            SECTIONS.get(q["gold_section"], q["gold_section"]), q.get("source_text", "")])


def apply_review_csv(testset: list[dict], path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        drop = {row["qid"] for row in csv.DictReader(f) if str(row.get("keep", "1")).strip() in ("0", "no", "n")}
    return [q for q in testset if q["qid"] not in drop]
