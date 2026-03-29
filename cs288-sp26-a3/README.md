# Berkeley EECS RAG Question-Answering System

**CS 288 (Natural Language Processing), UC Berkeley — Spring 2026**
Thomas Lee · Ziyu Ge · Shizhe Zhang

---

## Overview

An end-to-end Retrieval-Augmented Generation (RAG) pipeline for answering factual questions about UC Berkeley's EECS department. The system crawls the live EECS web presence, cleans and indexes the corpus, and serves answers via a hybrid retriever + LLM generator.

**Final evaluation (test set): 61.00% F1 / 88% Retrieval Recall**

---

## Pipeline

```
Async Web Crawler  -->  LLM-based Cleaner  -->  Semantic Chunker
                                                       |
                                                  FAISS Index (BGE)
                                                  BM25 Index
                                                       |
                    HyDE query expansion  -->  Hybrid Retrieval (RRF)
                                                       |
                                            Cross-Encoder Re-ranking
                                            + Keyword Boosting
                                                       |
                                            Llama-3.1-8B-Instruct
                                                       |
                                               Final Answer
```

### Key design choices

| Stage | Choice | Rationale |
|---|---|---|
| Crawling | Async + Global WAF Lock | Legacy `www2` subdomain aggressively rate-limits (HTTP 429); pausing all workers on rate-limit prevents IP bans |
| Embedding | `BAAI/bge-base-en-v1.5` | Outperforms `all-MiniLM-L12-v2` on domain-specific recall within the 400 MB model size constraint |
| Dense query | HyDE (Hypothetical Document Embeddings) | Bridges the gap between question phrasing and answer phrasing in embedding space |
| Sparse query | BM25 on original query (top-40) | Complements dense retrieval for exact entity and course-code matching |
| Fusion | Reciprocal Rank Fusion (RRF) | Parameter-free combination of dense and sparse ranked lists |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Semantic reranking over top-25 RRF candidates |
| Boosting | Keyword boost on title/URL | +0.5 score when query contains person names (`John Doe`) or course codes (`CS188`) that match chunk metadata |
| Generator | Llama-3.1-8B-Instruct via OpenRouter | Strict extractive prompt — must output the shortest possible span; "I don't know" only if context contains absolutely no relevant information |
| Top-k | k = 20 | 99.8% of peak F1 at k = 12 with lower latency; confirmed by ablation |

---

## Results

### Ablation: from baseline to final submission

> From the project report (Appendix: From Early Milestone to Final Submission)

| Milestone | Dev F1 | Recall | Test F1 | Key Change |
|---|---|---|---|---|
| Baseline | 47.18% | 67% | — | Initial system, noisy corpus |
| Corpus Cleanup | 59.21% | 79% | — | Clean corpus, index rebuild |
| k Tuning + Prompt | 62.26% | 80% | 55.76% | k=12, extractive prompt rules |
| Deduplication + Pool | 59.95% | 81% | 55.03% | Dedup, wider candidate pool |
| **BGE + HyDE (final)** | **62.66%** | **88%** | **61.00%** | BGE-base embeddings, HyDE, k=20 |

### Model selection (k = 10)

> From Table 1 of the report

| Model | EM | F1 |
|---|---|---|
| `llama-3.1-8b-instruct` | **0.5446** | **0.7048** |
| `allenai/olmo-3-7b` | 0.4732 | 0.6126 |
| `qwen/qwen3-8b` | 0.3482 | 0.5147 |
| `mistralai/mistral-7b-instruct` | 0.0000 | 0.0000 |

`llama-3.1-8b-instruct` outperformed all alternatives by ~14.5% F1. `mistral-7b` failed entirely due to silent deprecation on the OpenRouter API.

### Top-k sensitivity (llama-3.1-8b-instruct)

> From Table 2 of the report

| k | 6 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|
| F1 | 0.6630 | 0.6987 | 0.7048 | **0.7060** | 0.6969 |

F1 peaks at k = 12 and drops at k = 14, confirming the "lost in the middle" phenomenon as context noise accumulates. k = 10 captures 99.8% of peak F1 with 16% fewer context tokens.

### Error analysis (36 zero-F1 answers on hidden dev set)

> From Table 3 of the report

| Class | Count | Cause |
|---|---|---|
| Hallucination on Context | 12 | Retrieved correctly but model selects wrong value (competing numbers, wrong list member) |
| Retrieval Miss | 10 | Correct chunk absent from retrieved context; model hallucinates or refuses |
| Over-Refusal | 10 | Retrieval succeeded but model outputs "I don't know" — common on counting questions |
| Metric False Negative | 4 | Semantically correct but token-level F1 = 0 due to form differences (e.g., `510-642-3214` vs `1(510) 642-3214`) |

---

## Project Structure

```
cs288-sp26-a3/
├── data/
│   ├── raw/                    # Raw scraped corpora
│   ├── cleaned/                # Output of async cleaner
│   └── reference.jsonl         # Gold Q&A pairs for evaluation
├── models/
│   └── retrieval/              # Pre-built index (FAISS + BM25)
│       ├── index.faiss
│       ├── bm25.pkl
│       ├── chunks.json
│       └── metadata.json
├── src/
│   ├── crawler/                # Async EECS web scrapers
│   ├── processing/             # LLM-based cleaner & deduplicator
│   ├── retrieval/
│   │   ├── build_index.py      # Build FAISS + BM25 index
│   │   └── retrieve.py         # Hybrid retriever
│   ├── generation/
│   │   └── generator.py        # LLM answer generator
│   └── llm.py                  # OpenRouter API wrapper
├── experiments/
│   ├── evaluate_rag.py         # Evaluation script (EM / F1 / Recall)
│   └── eval_notes.md           # Ablation logs
├── report/
│   └── CS288_a3_report.pdf     # Full written report
├── main.py                     # Batch prediction entrypoint
├── run.sh                      # Autograder entrypoint
├── MANUAL.md                   # Full usage guide
└── requirements.txt
```

---

## Quick Start

### 1. Environment

```bash
conda create -n cs288_a3 python=3.10.12 -y
conda activate cs288_a3
pip install -r requirements.txt
export OPENROUTER_API_KEY="sk-or-..."
```

### 2. Build the index (if not pre-built)

```bash
# Clean raw corpus
python -m src.processing.async_cleaner \
    --input  data/raw/corpus_raw5.jsonl \
    --output data/cleaned/corpus_clean_v5.jsonl

# Build FAISS + BM25 index
python -m src.retrieval.build_index \
    --data   data/cleaned/corpus_clean_v5.jsonl \
    --output models/retrieval
```

### 3. Run predictions

```bash
python main.py data/questions.txt predictions.txt
```

### 4. Evaluate

```bash
python experiments/evaluate_rag.py \
    --reference data/reference.jsonl \
    --index models/retrieval \
    --k 20
```

See [`MANUAL.md`](MANUAL.md) for the full reference including chunking parameters, troubleshooting, and Gradescope submission instructions.

---

## Dependencies

| Package | Purpose |
|---|---|
| `faiss-cpu` | Dense vector index |
| `sentence-transformers` | BGE embeddings + cross-encoder re-ranking |
| `rank-bm25` | Sparse BM25 retrieval |
| `torch`, `transformers` | Model backend |
| `aiohttp`, `beautifulsoup4` | Web crawling (scraping only) |

---

## Contributions

- **Thomas:** Async web scraper, WAF bypass, retrieval corpus pipeline, ablation studies, report.
- **Ziyu:** QA dataset design, error analysis, first report draft.
- **Shizhe:** Project infrastructure, core RAG pipeline implementation, prompt engineering and HyDE design.
