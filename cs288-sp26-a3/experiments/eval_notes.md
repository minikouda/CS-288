# Evaluation Results Log

## corpus_clean_v5 — models/retrieval_clean — k=10

| Metric | Score |
|---|---|
| Exact Match | 53.57% |
| F1 Score | 70.38% |
| Retrieval Recall | 81.25% |

- Date: 2026-03-18
- Corpus: `data/cleaned/corpus_clean_v5.jsonl` (11,349 docs, 21,108 chunks)
- Index: `models/retrieval_clean/`
- Embedding model: `all-MiniLM-L12-v2`
- LLM: `meta-llama/llama-3.1-8b-instruct`
- Retrieval: hybrid dense + BM25 + RRF + cross-encoder rerank + keyword boost
- Reference: `data/reference.jsonl`

---

## corpus_clean_v5 — models/retrieval_clean — k=3

| Metric | Score |
|---|---|
| Exact Match | 45.54% |
| F1 Score | 61.79% |
| Retrieval Recall | 77.68% |

- Date: 2026-03-18
- Corpus: `data/cleaned/corpus_clean_v5.jsonl` (11,349 docs, 21,108 chunks)
- Index: `models/retrieval_clean/`
- Embedding model: `all-MiniLM-L12-v2`
- LLM: `meta-llama/llama-3.1-8b-instruct`
- Retrieval: hybrid dense + BM25 + RRF + cross-encoder rerank + keyword boost
- Reference: `data/reference.jsonl`
- Note: k=3 drops EM by ~8pp and recall by ~3.5pp vs k=10 — k=10 is clearly better