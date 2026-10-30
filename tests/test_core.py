import pytest

from drug_rag.answer import NOT_FOUND, parse_citations
from drug_rag.chunking import chunk_label
from drug_rag.embeddings import make_embedder, make_reranker
from drug_rag.fda import clean_text, dedupe_by_generic, normalize_label
from drug_rag.index import index_dir
from drug_rag.judge import ParseError, extract_json
from drug_rag.retrieve import Retriever, rrf


def test_normalize_label_from_openfda_shape():
    raw = {
        "set_id": "abc", "id": "v1", "effective_time": "20240102",
        "openfda": {"generic_name": ["WARFARIN SODIUM"], "brand_name": ["COUMADIN"], "route": ["ORAL"]},
        "indications_and_usage": ["1 INDICATIONS AND USAGE  Warfarin is indicated for   prophylaxis of thrombosis."],
        "contraindications": ["4 CONTRAINDICATIONS Pregnancy, except with mechanical heart valves. Bleeding tendencies."],
        "boxed_warning": ["WARNING: BLEEDING RISK. Warfarin can cause major or fatal bleeding."],
        "spl_unclassified_section": ["ignored"],
    }
    lab = normalize_label(raw)
    assert lab["generic_name"] == "Warfarin Sodium" and lab["brand_name"] == "Coumadin"
    assert set(lab["sections"]) == {"indications_and_usage", "contraindications", "boxed_warning"}
    assert "  " not in lab["sections"]["indications_and_usage"]


def test_normalize_label_rejects_unusable():
    assert normalize_label({"openfda": {}, "indications_and_usage": ["x" * 100]}) is None
    assert normalize_label({"openfda": {"generic_name": ["X"]}, "contraindications": ["y" * 100]}) is None


def test_dedupe_keeps_latest():
    a = {"generic_name": "Drug", "effective_time": "20200101", "v": 1}
    b = {"generic_name": "DRUG", "effective_time": "20240101", "v": 2}
    assert [d["v"] for d in dedupe_by_generic([a, b])] == [2]


def test_clean_text():
    assert clean_text("a  \t b\n\n  c\xa0d") == "a b\nc d"


def test_chunks_stay_in_section_and_carry_header(labels):
    chunks = chunk_label(labels[0], 200)
    assert len(chunks) > len(labels[0]["sections"])
    for c in chunks:
        assert c.text.startswith("Warfarin Sodium (Coumadin) | ")
        assert c.chunk_id.startswith(f"{c.set_id}:{c.section}:")
        assert len(c.text.split("\n", 1)[1]) <= 200


def test_rrf_prefers_items_ranked_well_in_both():
    fused = [i for i, _ in rrf([[1, 2, 3], [3, 1, 4]])]
    assert fused[0] == 1 and set(fused) == {1, 2, 3, 4}


def test_retrieval_modes_find_the_right_drug(built):
    root, _ = built
    r = Retriever(index_dir(300, root / "data" / "index"), make_embedder("hash"), make_reranker("hash"))
    for mode in ("bm25", "dense", "hybrid", "hybrid_rerank"):
        hits = r.search("What are the contraindications for metformin?", k=3, mode=mode)
        assert hits[0].rank == 1 and len(hits) == 3
        assert any(h.chunk.generic_name == "Metformin Hydrochloride" for h in hits), mode
    top = r.search("metformin contraindications renal impairment", k=1, mode="hybrid_rerank")[0]
    assert top.chunk.section == "contraindications"


def test_retriever_rejects_mismatched_embedder(built):
    root, _ = built

    class Other:
        name = "other"

    with pytest.raises(ValueError, match="built with"):
        Retriever(index_dir(300, root / "data" / "index"), Other())


def test_parse_citations():
    assert parse_citations("A [1]. B [2][7].", 5) == ([1, 2], [7])


def test_extract_json():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ParseError):
        extract_json("nope")


def test_not_found_constant_is_a_sentence():
    assert NOT_FOUND.endswith(".")


def test_drug_key_merges_salt_forms():
    from drug_rag.evaluate import drug_key

    assert drug_key("Duloxetine Hydrochloride") == drug_key("Duloxetine") == "duloxetine"
    assert drug_key("Bupivacaine Hydrochloride") == "bupivacaine"
    assert drug_key("Naproxen Sodium") == "naproxen"
    assert drug_key("Sodium Chloride") == "sodium chloride"
    assert drug_key("Magnesium Sulfate") != drug_key("Magnesium Citrate")
    assert drug_key("Potassium Chloride") != drug_key("Sodium Chloride")
