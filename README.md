# FDA Drug Label QA: hybrid RAG with a measured retrieval pipeline

Ask a question about a prescription drug and get a short answer that cites the exact FDA label
excerpts it came from. The project's focus is **measurement**: four retrievers and two chunk sizes
are compared on the same test set, and answers are graded for faithfulness (hallucination) and
correctness by an LLM judge.

- **Data:** ~1,200 prescription drug labels from the [openFDA drug label API](https://open.fda.gov/apis/drug/label/),
  one label per generic name, 16 clinical sections (indications, contraindications, boxed warning, dosage,
  interactions, adverse reactions, ...).
- **Retrieval:** BM25 keyword search, dense embeddings (`BAAI/bge-small-en-v1.5`, FAISS), hybrid fusion
  (reciprocal rank fusion), and a cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`). Models run on CPU
  through ONNX ([fastembed](https://github.com/qdrant/fastembed)), so no GPU is needed anywhere.
- **Generation:** any OpenAI-compatible API, Anthropic, or a free local model through Ollama. Answers must
  cite excerpts as `[1]`, `[2]` and refuse when the excerpts don't contain the answer.
- **Evaluation:** recall@1/5/10 and MRR for every chunk size × retriever; LLM-judge faithfulness and
  correctness for generated answers; everything logged to MLflow.
- **Serving:** FastAPI REST API (`/search`, `/ask`, `/health`) packaged with Docker; Gradio demo;
  a [live answer explorer](https://krishnagiri1.github.io/drug-rag/) on GitHub Pages.
- **Cost:** zero. Everything runs locally (Ollama on a laptop RTX 4080) and the demo is a static page.

> For information and research only. Not medical advice.

## Results

1,200 prescription drug labels (scanned from 4,122 openFDA records) → **55,859** chunks at 1,000 characters
(**112,211** at 500). Generator and judge: `qwen2.5:14b` running locally through Ollama. Full tables in
[`results/report.md`](results/report.md); every run is also logged to MLflow.

### Key findings

1. **Hybrid retrieval with a reranker finds the right passage far more often.** Recall@5 rises from
   **0.76** (BM25 keyword search) to **0.95** (hybrid + cross-encoder rerank), and recall@1 from 0.43 to **0.72**.
2. **Better retrieval gives better answers.** With the reranked pipeline the correct passage is retrieved for
   95% of questions (vs. 88% for dense-only), answers are fully correct 80% of the time (vs. 75%), and the
   model refuses less often (3% vs. 7%).
3. **The model stays grounded.** About **96%** of answer claims are supported by the cited excerpts in both
   setups, and 100% of answers cite sources with no invalid citations. Failures come mostly from retrieval
   misses (leading to refusals), not from hallucination.
4. **Keyword search breaks on simple questions.** On template questions like "What are the contraindications
   for X?", BM25 reaches only **0.64** recall@5, because drug names and section words appear across hundreds of
   other labels. Dense and reranked retrieval reach 0.88–0.94.
5. **The reranker costs latency.** Dense search alone reaches 0.88–0.91 recall@5 in ~25 ms; the reranker adds
   ~1 s per query on a laptop CPU for the last few points. Chunk size (500 vs. 1,000 characters) barely matters.

### Retrieval (152 questions: 102 LLM-written and human-reviewed, 50 template)

| Retriever | Chunk size | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Latency |
|---|---|---|---|---|---|---|
| BM25 (keyword) | 500 | 0.43 | 0.76 | 0.86 | 0.57 | 233 ms |
| BM25 (keyword) | 1,000 | 0.51 | 0.79 | 0.90 | 0.62 | 138 ms |
| Dense (bge-small + FAISS) | 500 | 0.68 | 0.91 | 0.93 | 0.78 | 36 ms |
| Dense (bge-small + FAISS) | 1,000 | 0.64 | 0.88 | 0.92 | 0.75 | 23 ms |
| Hybrid (BM25 + dense, RRF) | 500 | 0.64 | 0.91 | 0.93 | 0.75 | 329 ms |
| Hybrid (BM25 + dense, RRF) | 1,000 | 0.59 | 0.93 | 0.96 | 0.72 | 211 ms |
| Hybrid + cross-encoder rerank | 500 | 0.67 | 0.94 | 0.96 | 0.78 | 1.08 s |
| **Hybrid + cross-encoder rerank** | **1,000** | **0.72** | **0.95** | **0.97** | **0.82** | 1.22 s |

A hit is a retrieved chunk from the right drug and the right label section; salt/form variants of one drug
("Duloxetine" / "Duloxetine Hydrochloride") count as the same drug. Requiring the exact label gives 0.95 for the
best setup as well.

### Answer quality (100 LLM-written questions, k = 5, 1,000-character chunks)

| Retrieval | Right passage retrieved | Claims supported | Correctness | Fully correct | Refused | Median latency |
|---|---|---|---|---|---|---|
| Dense only | 88% | 96% | 0.82 | 75% | 7% | 3.8 s |
| **Hybrid + rerank** | **95%** | 96% | **0.87** | **80%** | **3%** | 5.0 s |

### What I'd try next

- A stronger, separate judge model: here the same 14B model writes and grades answers, which may be lenient
  (my [agent-judge](https://github.com/krishnagiri1/agent-judge) project measured this model inflating scores).
- A human check of 20–30 faithfulness verdicts to measure judge reliability directly.
- Metadata filtering by drug name before search, which should fix most keyword-search failures cheaply.

## How it works

```
openFDA labels ──► normalize + dedupe (1 label per generic) ──► section-aware chunks
                                                                 "Warfarin (Coumadin) | Contraindications\n..."
                                                                       │
                                     ┌─────────────────────────────────┴───────────────┐
                                     ▼                                                 ▼
                              BM25 (rank_bm25)                          bge-small embeddings → FAISS
                                     └──────────── reciprocal rank fusion ─────────────┘
                                                          │ top 30
                                                          ▼
                                            cross-encoder rerank → top k excerpts
                                                          │
                                                          ▼
                                     LLM answer with [n] citations, or a refusal
```

**Chunking** (`drug_rag/chunking.py`): chunks never cross section boundaries, and each one starts with a
header naming the drug and section. Without that header, a passage like "Do not exceed 4 g per day" can't
be matched to its drug. Splitting inside a section uses LangChain's `RecursiveCharacterTextSplitter`.

**Hybrid retrieval** (`drug_rag/retrieve.py`): BM25 wins on exact names and rare terms, embeddings win on
paraphrase. Reciprocal rank fusion merges the two ranked lists without calibrating their score scales. The
cross-encoder reads query and passage together (more accurate, slower), so it only re-scores the top 30.

**Test set** (`drug_rag/testset.py`): template questions ("What are the contraindications for X?") plus
LLM-written questions about one specific passage each, with a reference answer. Gold is the (label, section),
so indexes with different chunk sizes are scored fairly. Generated questions are exported to a CSV for a
quick human spot-check before use.

**Answer grading** (`drug_rag/judge.py`): an LLM judge splits each answer into claims and checks each claim
against the retrieved excerpts (faithfulness), and compares the answer with the reference (correctness).
It uses the same approach as my [agent-judge](https://github.com/krishnagiri1/agent-judge) project:
temperature 0, strict JSON, parse retries, cached calls.

## Setup

Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest -q                     # 19 tests, no downloads or API keys needed
```

For the LLM: either install [Ollama](https://ollama.com) and `ollama pull qwen2.5:14b` (free, local), or set
`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`.

## Run it

```bash
# 1. Download labels (~1-3 min). Optional: set FDA_API_KEY for higher rate limits.
python -m drug_rag download --n 1200

# 2. Chunk + embed + index, for both chunk sizes (~5-15 min on a laptop CPU)
python -m drug_rag index

# 3. Build the test set: 50 template + 120 LLM-written questions, then spot-check
python -m drug_rag testset --provider ollama --model qwen2.5:14b
#    open data/testset.review.csv, set keep=0 on bad questions, save, then:
python -m drug_rag apply-review

# 4. Retrieval evaluation: 2 chunk sizes x 4 retrievers (no LLM needed, minutes)
python -m drug_rag eval-retrieval

# 5. Answer evaluation on the generated questions (LLM generator + LLM judge; run overnight locally)
python -m drug_rag eval-answers --provider ollama --model qwen2.5:14b --mode hybrid_rerank
python -m drug_rag eval-answers --provider ollama --model qwen2.5:14b --mode dense       # baseline

# 6. Report + MLflow
python -m drug_rag report
mlflow ui

# 7. Try it
python -m drug_rag ask "What are the contraindications for metformin?" --provider ollama --model qwen2.5:14b
python -m drug_rag demo --provider ollama --model qwen2.5:14b         # Gradio at http://127.0.0.1:7860
```

A stronger judge than the generator gives more trustworthy grades: add `--judge-provider openai
--judge-model <model>` to `eval-answers`. Every LLM call is cached in `.cache/`, and `eval-answers` resumes
where it stopped.

## The API

```powershell
# locally, in PowerShell (the server reads its LLM settings from environment variables)
$env:LLM_PROVIDER = "ollama"; $env:LLM_MODEL = "qwen2.5:14b"
python -m drug_rag serve                                      # http://127.0.0.1:8000/docs
```

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | – | index info, retriever, LLM |
| POST | `/search` | `{"question": "...", "k": 5, "mode": "hybrid_rerank"}` | ranked excerpts (no LLM needed) |
| POST | `/ask` | same | answer, refusal flag, sources with `cited` flags, latency |

## Docker

See [docs/docker.md](docs/docker.md). The image bundles the code, both ONNX models, and the prebuilt index;
the container can use the Ollama model running on the host, so `/ask` works fully offline and free:

```bash
docker build -t drug-rag .
docker run --rm -p 8000:8000 -e LLM_PROVIDER=ollama -e LLM_MODEL=qwen2.5:14b \
  -e LLM_BASE_URL=http://host.docker.internal:11434 drug-rag
```

## Live answer explorer (GitHub Pages)

```bash
python -m drug_rag export-site      # writes docs/index.html + docs/data.js from your results
```

Commit and push, then on GitHub: **Settings → Pages → Deploy from a branch → `main` / `/docs`**.

## Project layout

```
drug_rag/
  fda.py         openFDA download, normalization, de-duplication
  chunking.py    section-aware chunks with drug/section headers
  embeddings.py  fastembed (ONNX) embedder + cross-encoder; hash stand-ins for tests
  index.py       FAISS + chunk store per chunk size
  retrieve.py    BM25, dense, RRF hybrid, reranking
  answer.py      cited answer generation and citation parsing
  judge.py       LLM-judge faithfulness and correctness
  testset.py     template + LLM-generated questions, review CSV
  evaluate.py    retrieval metrics and answer grading (resumable)
  report.py      results/report.md + MLflow logging
  site.py        static answer explorer for GitHub Pages
  api.py         FastAPI service
  ui.py          Gradio demo
  llm.py         OpenAI-compatible / Anthropic / Ollama / mock clients with a SQLite cache
Dockerfile, requirements-serve.txt   API image (see docs/docker.md)
docs/                                Docker guide + the published answer explorer
app.py                               Hugging Face Spaces entry point
tests/                               unit + end-to-end tests (no network)
```

## Limitations

- Gold is (label, section). A question about a risk mentioned in both "Boxed Warning" and "Warnings and
  Precautions" counts as a miss if only the other section is retrieved, so recall is slightly pessimistic.
- Generated questions are written by an LLM from one passage, which favors retrievers that match the
  passage's wording; template questions and the human spot-check offset this partly.
- The same 14B model generates and grades answers, so faithfulness and correctness may be optimistic.
- 1 of 100 dense-only answer requests failed twice with a local Ollama server error, so that run has n = 99.
- Generated questions were written by an LLM from single passages and then reviewed by hand (18 of 120 removed);
  they may still favor retrievers that match the passage's wording.
- 1,200 labels is a subset of openFDA's prescription labels; results on rarer drugs may differ.
- One label per generic name: strengths, formulations and manufacturers can differ between labels.
