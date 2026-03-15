# RAG System Implementation Status

## Overview
This document summarizes the steps taken to implement the Retrieval-Augmented Generation (RAG) system for Assignment 3, along with a checklist of completed and pending items.

## Completed Tasks

*   **Data Processing & Indexing (`src/retrieval/build_index.py`)**
    *   Parsed and chunked the provided processed JSON files.
    *   Implemented a simple sliding window chunking strategy.
    *   Combined document title, URL, and content into chunks for better context.
    *   Built a Dense Retrieval index using FAISS and `sentence-transformers/all-MiniLM-L6-v2`.
    *   Built a Sparse Retrieval index using BM25 (`rank-bm25`).

*   **Retrieval Component (`src/retrieval/retrieve.py`)**
    *   Implemented a `Retriever` class that loads the FAISS index, BM25 model, and chunk metadata.
    *   Created a hybrid retrieval function that combines top-K results from both Dense (FAISS) and Sparse (BM25) methods to improve recall for both semantic and exact keyword matching.

*   **Generation Component (`src/generation/generator.py`)**
    *   Developed a `Generator` class to format the retrieved context and user query into a prompt.
    *   Integrated with the provided `src/llm.py` wrapper to call the OpenRouter API using the allowed `meta-llama/llama-3.1-8b-instruct` model.

*   **Pipeline Integration (`main.py` & `run.sh`)**
    *   Created `main.py` to orchestrate the entire pipeline: reading questions, retrieving context, generating answers, and writing predictions.
    *   Wrote `run.sh` to meet the autograder requirements (`bash run.sh <questions_txt> <predictions_out>`).

## Checklist

- [x] Read and understand PDF requirements.
- [x] Process `data/processed/*.json` files.
- [x] Implement document chunking.
- [x] Build FAISS Dense index (<400MB embedding model).
- [x] Build BM25 Sparse index.
- [x] Implement Hybrid Retrieval function.
- [x] Implement LLM prompt generation and integration with `llm.py`.
- [x] Create main execution script (`main.py`).
- [x] Create bash entry point (`run.sh`).
- [ ] **Pending:** Set `OPENROUTER_API_KEY` in the environment to execute the LLM.
- [ ] **Pending:** Run official evaluation on the hidden dev set and perform ablations.
- [ ] **Pending:** Write the final report.

## Environment Management
*   **Conda Environment:** Created `rag_env` with Python 3.10.12.
*   **Activation:** Use `conda activate rag_env` to run the project locally.
*   **Dependencies:** All dependencies from `requirements.txt` are installed in this environment.
