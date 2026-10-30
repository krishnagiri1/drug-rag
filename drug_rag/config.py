"""Shared defaults. Every value can be overridden from the command line or environment."""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DRUG_RAG_DATA", "data"))
LABELS_PATH = DATA_DIR / "labels.jsonl"
INDEX_DIR = DATA_DIR / "index"
TESTSET_PATH = DATA_DIR / "testset.jsonl"
RESULTS_DIR = Path(os.environ.get("DRUG_RAG_RESULTS", "results"))

# Small, fast models that run on CPU (ONNX via fastembed), so the same index works on a
# laptop, in Docker, and on a small cloud instance.
EMBED_MODEL = os.environ.get("DRUG_RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.environ.get("DRUG_RAG_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")

# Chunk sizes compared in the evaluation (characters, roughly 4 characters per token).
CHUNK_SIZES = (500, 1000)
DEFAULT_CHUNK_SIZE = int(os.environ.get("DRUG_RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP_RATIO = 0.15

# Retrieval
RETRIEVAL_MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")
DEFAULT_MODE = os.environ.get("DRUG_RAG_MODE", "hybrid_rerank")
CANDIDATES = 30  # how many hybrid candidates the reranker re-scores
RRF_K = 60  # reciprocal rank fusion constant (the standard value from Cormack et al., 2009)

# Label sections we index, in display order. Prescription labels use the first set;
# older labels use `warnings` / `precautions` instead of `warnings_and_cautions`.
SECTIONS: dict[str, str] = {
    "boxed_warning": "Boxed Warning",
    "indications_and_usage": "Indications and Usage",
    "dosage_and_administration": "Dosage and Administration",
    "dosage_forms_and_strengths": "Dosage Forms and Strengths",
    "contraindications": "Contraindications",
    "warnings_and_cautions": "Warnings and Precautions",
    "warnings": "Warnings",
    "precautions": "Precautions",
    "adverse_reactions": "Adverse Reactions",
    "drug_interactions": "Drug Interactions",
    "use_in_specific_populations": "Use in Specific Populations",
    "pregnancy": "Pregnancy",
    "pediatric_use": "Pediatric Use",
    "geriatric_use": "Geriatric Use",
    "overdosage": "Overdosage",
    "mechanism_of_action": "Mechanism of Action",
}
