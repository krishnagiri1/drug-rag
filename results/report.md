# Results

## Retrieval

152 test questions (102 generated, 50 template). A hit means a retrieved chunk comes from the right drug **and** the right section (salt/form variants such as "Duloxetine" and "Duloxetine Hydrochloride" count as the same drug). "Exact label" also requires the specific label the question was written from.

| Chunk size | Retriever | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Right drug in top 5 | Recall@5, exact label | Latency (ms/query) |
|---|---|---|---|---|---|---|---|---|
| 500 chars | BM25 (keyword) | 0.43 | 0.76 | 0.86 | 0.57 | 93% | 0.76 | 233 |
| 500 chars | Dense (embeddings) | 0.68 | 0.91 | 0.93 | 0.78 | 97% | 0.91 | 36 |
| 500 chars | Hybrid (BM25 + dense, RRF) | 0.64 | 0.91 | 0.93 | 0.75 | 99% | 0.88 | 329 |
| 500 chars | Hybrid + cross-encoder rerank | 0.67 | 0.94 | 0.96 | 0.78 | 97% | 0.92 | 1079 |
| 1000 chars | BM25 (keyword) | 0.51 | 0.79 | 0.90 | 0.62 | 96% | 0.78 | 138 |
| 1000 chars | Dense (embeddings) | 0.64 | 0.88 | 0.92 | 0.75 | 96% | 0.88 | 23 |
| 1000 chars | Hybrid (BM25 + dense, RRF) | 0.59 | 0.93 | 0.96 | 0.72 | 100% | 0.92 | 211 |
| 1000 chars | Hybrid + cross-encoder rerank | 0.72 | 0.95 | 0.97 | 0.82 | 99% | 0.95 | 1216 |

Best: **Hybrid + cross-encoder rerank, 1000-char chunks**, recall@5 0.76 → 0.95 vs. the weakest configuration (BM25 (keyword), 500 chars).

Recall@5 by question type:

| Chunk size | Retriever | generated | template |
|---|---|---|---|
| 500 | BM25 (keyword) | 0.82 | 0.64 |
| 500 | Dense (embeddings) | 0.91 | 0.92 |
| 500 | Hybrid (BM25 + dense, RRF) | 0.90 | 0.92 |
| 500 | Hybrid + cross-encoder rerank | 0.95 | 0.92 |
| 1000 | BM25 (keyword) | 0.86 | 0.64 |
| 1000 | Dense (embeddings) | 0.88 | 0.88 |
| 1000 | Hybrid (BM25 + dense, RRF) | 0.94 | 0.90 |
| 1000 | Hybrid + cross-encoder rerank | 0.95 | 0.94 |

## Answer quality (graded by an LLM judge)

Faithfulness = share of the answer's claims supported by the retrieved excerpts (hallucination check). Correctness = agreement with the test set's reference answer (correct 1, partial 0.5, incorrect 0).

| Config | Generator | Judge | n | Faithfulness | Fully faithful answers | Correctness | Refused | Answers with citations | Gold passage retrieved | Median latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 chars, Dense (embeddings), k=5 | ollama:qwen2.5:14b | ollama:qwen2.5:14b | 99 | 96% | 96% | 0.82 | 7% | 100% | 88% | 3.8 |
| 1000 chars, Hybrid + cross-encoder rerank, k=5 | ollama:qwen2.5:14b | ollama:qwen2.5:14b | 100 | 96% | 96% | 0.87 | 3% | 100% | 95% | 5.0 |
