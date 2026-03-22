# CS 288 A3 — RAG System Manual

A Retrieval-Augmented Generation (RAG) system for answering Berkeley EECS questions. This manual covers every step from raw data to running predictions and evaluating results.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Project Structure](#2-project-structure)
3. [Step 1 — Clean the Corpus](#3-step-1--clean-the-corpus)
4. [Step 2 — Build the Index](#4-step-2--build-the-index)
5. [Step 3 — Run Predictions](#5-step-3--run-predictions)
6. [Step 4 — Evaluate](#6-step-4--evaluate)
7. [Configuration Reference](#7-configuration-reference)
8. [Allowed LLM Models](#8-allowed-llm-models)
9. [Troubleshooting](#9-troubleshooting)
10. [Gradescope Submission Guide](#10-gradescope-submission-guide)

---

## 1. Prerequisites

### Creating the Conda Environment

There are two separate requirement sets in this project:

| File | Purpose |
|---|---|
| `requirements.txt` | RAG inference — needed by the autograder and for running/evaluating the model |
| `requirements_scraping.txt` | Data collection only — `aiohttp`, `beautifulsoup4`, `lxml` |

#### RAG environment (required for all Steps 1–4)

Create the environment once:

```bash
conda create -n cs288_a3 python=3.10.12 -y
conda activate cs288_a3
pip install -r requirements.txt
```

The autograder uses **Python 3.10.12** exactly. Match this locally to avoid version surprises.

#### Scraping extras (only needed to re-crawl the EECS website)

With `cs288_a3` active, install the additional scraping packages:

```bash
pip install -r requirements_scraping.txt
```

These packages (`aiohttp`, `beautifulsoup4`, `lxml`) are **not** included in `requirements.txt` because the autograder does not need them.

#### Activating the environment

Every time you open a new terminal, activate the environment before running any command:

```bash
conda activate cs288_a3
```

Verify the environment is correct:

```bash
python --version          # should print Python 3.10.x
python -c "import faiss; import sentence_transformers; import rank_bm25; print('OK')"
```

---

### API Key

The generation step calls OpenRouter to serve the LLM. Set your key before running any generation or evaluation:

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

To persist it across sessions, add the line above to your `~/.zshrc` or `~/.bashrc`.

### Working Directory

All commands below assume you are at the project root:

```bash
cd /path/to/cs288-sp26-a3
```

---

## 2. Project Structure

```
cs288-sp26-a3/
├── data/
│   ├── raw/                    # Raw scraped corpora (corpus_raw*.jsonl)
│   ├── cleaned/                # Output of async cleaner
│   └── reference.jsonl         # Gold Q&A pairs for evaluation
├── models/
│   └── retrieval/              # Built index (FAISS + BM25)
│       ├── index.faiss
│       ├── bm25.pkl
│       ├── chunks.json
│       └── metadata.json
├── src/
│   ├── processing/
│   │   └── async_cleaner.py    # Step 1: clean & deduplicate
│   ├── retrieval/
│   │   ├── build_index.py      # Step 2: build FAISS + BM25
│   │   └── retrieve.py         # Hybrid retriever (dense + BM25 + rerank)
│   ├── generation/
│   │   └── generator.py        # LLM-based answer generator
│   └── llm.py                  # OpenRouter API wrapper
├── experiments/
│   ├── evaluate_rag.py         # Step 4: evaluation script
│   └── last_eval_results.json  # Most recent eval output
└── main.py                     # Step 3: batch prediction entrypoint
```

---

## 3. Step 1 — Clean the Corpus

The async cleaner reads a raw JSONL corpus, removes boilerplate, normalizes whitespace, strips non-ASCII characters, drops documents shorter than 100 characters, and deduplicates by MD5 content hash.

### Basic usage

```bash
python -m src.processing.async_cleaner \
    --input  data/raw/corpus_raw5.jsonl \
    --output data/cleaned/corpus_clean_v5.jsonl
```

### Arguments

| Argument   | Default              | Description                                              |
|------------|----------------------|----------------------------------------------------------|
| `--input`  | `data/raw`           | Path to a `.jsonl` file **or** a directory of `.json` files |
| `--output` | auto-timestamped     | Output `.jsonl` path. Auto-generated as `data/cleaned/corpus_clean_<timestamp>.jsonl` if omitted |

### Processing a directory of JSON files

```bash
python -m src.processing.async_cleaner \
    --input  data/raw/ \
    --output data/cleaned/corpus_clean_all.jsonl
```

### What it produces

Each line in the output is a JSON object with the original fields plus:

- `content` — cleaned text
- `content_hash` — MD5 hex digest used for deduplication

---

## 4. Step 2 — Build the Index

Reads the cleaned JSONL, recursively chunks each document, encodes chunks with a SentenceTransformer embedding model, and saves a FAISS vector index and a BM25 index.

### Basic usage

```bash
python -m src.retrieval.build_index \
    --data   data/cleaned/corpus_clean_v5.jsonl \
    --output models/retrieval
```

### Arguments

| Argument   | Default                   | Description                                              |
|------------|---------------------------|----------------------------------------------------------|
| `--data`   | `data/processed`          | Input: a `.jsonl` file or a directory of `.json` files   |
| `--output` | `models/retrieval`        | Directory where index files are written                  |
| `--model`  | `all-MiniLM-L12-v2`       | SentenceTransformer model name (from HuggingFace)        |

> **Note:** If a directory `data/augmented/` exists and is non-empty, the script automatically uses it instead of `--data`. Remove or empty that directory to force use of the specified input.

### Output files

| File             | Contents                                      |
|------------------|-----------------------------------------------|
| `index.faiss`    | FAISS flat L2 index of chunk embeddings       |
| `bm25.pkl`       | Serialized BM25Okapi object                   |
| `chunks.json`    | List of chunk text strings                    |
| `metadata.json`  | Per-chunk metadata: `url`, `title`, `file`, `type` |

### Chunking parameters (edit `build_index.py` to change)

| Parameter    | Default | Description                        |
|--------------|---------|------------------------------------|
| `chunk_size` | 1000    | Maximum characters per chunk       |
| `overlap`    | 200     | Character overlap between chunks   |

---

## 5. Step 3 — Run Predictions

`main.py` reads a list of questions (one per line), runs the full RAG pipeline for each, and writes one answer per line to the output file.

### Usage

```bash
python main.py <questions_file> <predictions_file>
```

### Example

```bash
python main.py data/questions.txt predictions.txt
```

### What happens internally

1. **Retriever** loads the index from `models/retrieval/`.
2. For each question:
   - **Dense retrieval** — top-30 chunks via FAISS cosine similarity
   - **BM25 retrieval** — top-30 chunks via keyword scoring
   - **RRF fusion** — Reciprocal Rank Fusion merges both lists
   - **Cross-encoder reranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2` rescores top-20 candidates
   - **Keyword boosting** — bumps chunks whose title/URL match person names or course codes in the query
   - **Generator** sends top-5 chunks to the LLM and returns a short answer
3. Answers are written one per line (newlines stripped).

---

## 6. Step 4 — Evaluate

Runs the full RAG pipeline against a reference file and reports Exact Match, F1, and Retrieval Recall.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--reference` | `data/hidden_dev.jsonl` | Path to first (or only) reference JSONL file |
| `--reference2` | _(none)_ | Path to second reference JSONL file; when provided, both are evaluated and combined totals are reported |
| `--index` | `models/retrieval` | Path to the retrieval index directory |
| `--k` | `10` | Number of chunks to retrieve per query |

### Usage examples

```bash
# Default — hidden_dev with the standard index
python experiments/evaluate_rag.py

# Single file, custom index
python experiments/evaluate_rag.py \
    --reference data/reference.jsonl \
    --index models/retrieval_bs

# Both reference sets, new corpus, report combined totals
python experiments/evaluate_rag.py \
    --reference data/reference.jsonl \
    --reference2 data/hidden_dev.jsonl \
    --index models/retrieval_bs

# Increase retrieved chunks
python experiments/evaluate_rag.py \
    --reference data/hidden_dev.jsonl \
    --index models/retrieval \
    --k 15
```

When `--reference2` is provided the script prints per-file metrics followed by combined totals across all samples.

### Reference file format

Each line is a JSON object:

```json
{"question": "Who chairs the CS division?", "answer": "John Doe", "url": "https://eecs.berkeley.edu/..."}
```

- `answer` may contain multiple acceptable answers separated by `|` (e.g. `"John Doe|J. Doe"`).
- `url` is optional but used for retrieval recall calculation.

### Metrics

| Metric            | Description                                                        |
|-------------------|--------------------------------------------------------------------|
| Exact Match (EM)  | Prediction matches any gold answer after normalization             |
| F1 Score          | Token-level overlap between prediction and best-matching gold answer |
| Retrieval Recall  | Fraction of queries where the gold URL or gold answer text appeared in retrieved chunks |

Normalization removes punctuation, articles (a/an/the), and lowercases before comparison.

### Output

Results are printed to stdout and saved to `experiments/last_eval_results.json`:

```json
{
  "overall_em": 0.55,
  "overall_f1": 0.63,
  "details": [
    {
      "question": "...",
      "ground_truths": ["..."],
      "prediction": "...",
      "em": 1,
      "f1": 1.0,
      "retrieval_recall": true,
      ...
    }
  ]
}
```

> **Note:** The default index is `models/retrieval`. Use `--index` to point at any other built index (e.g. `models/retrieval_bs`).

---

## 7. Configuration Reference

| Location                          | Variable / Line       | What it controls                         |
|-----------------------------------|-----------------------|------------------------------------------|
| `main.py` line 16                 | `Retriever("models/retrieval")` | Index used for batch predictions  |
| `main.py` line 17                 | `Generator(model=...)` | LLM model for predictions               |
| `experiments/evaluate_rag.py` | `--index` argument (default `models/retrieval`) | Index used for evaluation |
| `experiments/evaluate_rag.py` | `--reference` / `--reference2` arguments | Reference file(s) for evaluation |
| `src/retrieval/retrieve.py` line 47 | `k=10` default       | Number of final chunks returned          |
| `src/retrieval/build_index.py` line 127 | `chunk_size`, `overlap` | Chunking parameters               |

---

## 8. Allowed LLM Models

The following models are available via OpenRouter (defined in `src/llm.py`):

| Model ID                            | Notes                  |
|-------------------------------------|------------------------|
| `meta-llama/llama-3.1-8b-instruct`  | Default                |
| `meta-llama/llama-3-8b-instruct`    |                        |
| `qwen/qwen3-8b`                     |                        |
| `qwen/qwen-2.5-7b-instruct`         |                        |
| `allenai/olmo-3-7b-instruct`        |                        |
| `mistralai/mistral-7b-instruct`     |                        |

Pass the model ID string to `Generator(model=...)` or `call_llm(model=...)`.

---

## 9. Troubleshooting

**`ModuleNotFoundError: No module named 'faiss'`**
Make sure you are inside the conda environment: `conda activate cs288_a3`

**`ValueError: OPENROUTER_API_KEY environment variable is required`**
Export the key: `export OPENROUTER_API_KEY="sk-or-..."`

**`FileNotFoundError` when loading index**
The index directory must contain all four files (`index.faiss`, `bm25.pkl`, `chunks.json`, `metadata.json`). Re-run Step 2 if any are missing.

**Retriever uses `data/augmented/` unexpectedly**
`build_index.py` auto-selects `data/augmented/` when it exists and is non-empty. Either remove that directory or pass your intended `--data` path and temporarily rename `data/augmented/`.

**Slow embedding during index build**
This is normal on CPU — 21k chunks takes ~4-5 minutes. The model caches to `~/.cache/huggingface/` after first download.

---

## 10. Gradescope Submission Guide

### What to include (and exclude)

The grader only needs the code and the pre-built retrieval index. **Do not include** raw corpora, cleaned corpora, scraping code, or any other preprocessing artifacts — they are not needed at inference time and will bloat the zip.

**Include:**

```
run.sh                          ← required entrypoint (exact filename)
main.py                         ← batch prediction script
requirements.txt                ← fixed dependency list
src/
  __init__.py                   ← if present
  llm.py                        ← OpenRouter wrapper (do NOT modify)
  retrieval/
    __init__.py
    retrieve.py
  generation/
    __init__.py
    generator.py
models/
  retrieval/                    ← pre-built index
    index.faiss
    bm25.pkl
    chunks.json
    metadata.json
```

**Exclude:**

```
data/raw/                       ← raw scraped corpora (large, not needed)
data/cleaned/                   ← intermediate cleaned files
src/crawler/                    ← scraping code
src/processing/                 ← cleaning code
src/utils/                      ← utilities only used offline
experiments/                    ← evaluation scripts, not needed by grader
models/legacy/                  ← old index snapshots (if present)
*.pyc / __pycache__/            ← compiled bytecode
submission.zip                  ← don't nest zips
```

---

### Pre-submission checklist

#### 1. Verify `run.sh`

The autograder calls exactly:

```bash
bash run.sh <questions_txt_path> <predictions_out_path>
```

Your `run.sh` must use `python3` (not `python`):

```bash
#!/bin/bash
python3 main.py "$1" "$2"
```

All paths in the code are **relative** to the project root. Never use absolute paths like `/Users/yourname/...`.

#### 2. Do NOT modify `llm.py`

The grader **overwrites** `src/llm.py` with its own copy before running. Any changes you made to that file will be lost. All LLM calls must go through `call_llm()` from `src/llm.py` only — calling OpenRouter directly from any other file will result in a **score of 0**.

#### 3. Verify output format

- One prediction per line, same order as input questions.
- No newlines inside any prediction (the pipeline already calls `.replace("\n", " ")`).
- Total line count must equal the number of input questions.

#### 4. Check autograder constraints

| Constraint | Requirement |
|---|---|
| Python version | 3.10.12 |
| RAM | 4 GB max — no GPU |
| Embedding model size | 400 MB or less (`all-MiniLM-L12-v2` is ~90 MB, safe) |
| Per-question latency | ~0.6 s average (30-minute total time limit for 100 questions) |
| LLM access | OpenRouter only, via the provided `llm.py` |

---

### Building the zip

From the project root, run:

```bash
zip -r submission.zip \
  run.sh \
  main.py \
  requirements.txt \
  src/llm.py \
  src/__init__.py \
  src/retrieval/ \
  src/generation/ \
  models/retrieval/ \
  --exclude "**/__pycache__/*" \
  --exclude "**/*.pyc"
```

Verify the contents before uploading:

```bash
unzip -l submission.zip
```

Make sure `run.sh` is at the top level of the zip (not inside a subdirectory), and `models/retrieval/` contains all four index files.

---

### Submission limits

| Milestone | Deadline | Submission limit | Late days |
|---|---|---|---|
| Early milestone | 03/17 (Tue) 05:59 PM | 5 | Not allowed |
| Final | 03/19 (Thu) 05:59 PM | 5 | Up to 3 days |

The autograder evaluates 100 questions per submission. Budget your submission attempts — debug locally first using `experiments/evaluate_rag.py` before uploading.