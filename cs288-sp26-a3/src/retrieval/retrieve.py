import faiss
import json
import os
import numpy as np
import pickle
import re
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.llm import call_llm

def tokenize(text):
    return text.lower().split()

class Retriever:
    def __init__(self, index_dir, model_name="BAAI/bge-base-en-v1.5"):
        self.model = SentenceTransformer(model_name)
        # Tiny but powerful re-ranker (Fits in 4GB RAM)
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        self.index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        with open(os.path.join(index_dir, "bm25.pkl"), 'rb') as f:
            self.bm25 = pickle.load(f)
        with open(os.path.join(index_dir, "chunks.json"), 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        with open(os.path.join(index_dir, "metadata.json"), 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

    def retrieve_dense(self, query, k=5):
        query_vector = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vector, k)
        return [idx for idx in indices[0] if idx != -1]

    def retrieve_bm25(self, query, k=5):
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        return np.argsort(scores)[-k:][::-1].tolist()

    def keyword_boost(self, query, candidates_metadata):
        # Extract potential names (John Doe) or course codes (CS188)
        keywords = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b|\b[A-Z]{2,}\d{2,3}[A-Z]?\b', query)
        boost_scores = [0.0] * len(candidates_metadata)
        
        for i, meta in enumerate(candidates_metadata):
            for kw in keywords:
                if kw.lower() in meta['title'].lower() or kw.lower() in meta['url'].lower():
                    boost_scores[i] += 0.5 # Metadata match boost
        return boost_scores

    def hypothetical_answer(self, query):
        """HyDE: generate a short hypothetical answer to embed for dense retrieval."""
        try:
            return call_llm(
                query=f"Question: {query}\nWrite a short factual answer as if you know it. 1-2 sentences only.",
                system_prompt="You are a knowledgeable Berkeley EECS assistant. Answer concisely with facts.",
                model="meta-llama/llama-3.1-8b-instruct",
                max_tokens=64,
                temperature=0,
            )
        except Exception:
            return query  # fallback to original query on failure

    def retrieve(self, query, k=10):
        # 1. Broad Hybrid Retrieval
        # HyDE: embed a hypothetical answer for dense retrieval; BM25 uses original query
        hyp = self.hypothetical_answer(query)
        dense_indices = self.retrieve_dense(hyp, k=40)
        bm25_indices = self.retrieve_bm25(query, k=40)

        # 2. RRF Fusion
        rrf_scores = {}
        for rank, idx in enumerate(dense_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (60 + rank)
        for rank, idx in enumerate(bm25_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (60 + rank)

        # Select top 25 candidates for re-ranking
        candidate_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:25]
        candidate_texts = [self.chunks[idx] for idx in candidate_indices]
        candidate_meta = [self.metadata[idx] for idx in candidate_indices]

        # 3. Cross-Encoder Re-ranking (Semantic relevance)
        pairs = [[query, text] for text in candidate_texts]
        rerank_scores = self.reranker.predict(pairs)

        # 4. Keyword Boosting
        boosts = self.keyword_boost(query, candidate_meta)
        final_scores = rerank_scores + boosts

        # 5. Rank and deduplicate, then return top k unique chunks
        ranked_results = sorted(zip(candidate_indices, final_scores), key=lambda x: x[1], reverse=True)
        seen_content = set()
        final_indices = []
        for idx, score in ranked_results:
            content = self.chunks[idx]
            if content not in seen_content:
                seen_content.add(content)
                final_indices.append(idx)
            if len(final_indices) == k:
                break

        results = []
        for idx in final_indices:
            results.append({
                "content": self.chunks[idx],
                "metadata": self.metadata[idx],
                "id": idx
            })
        return results

if __name__ == "__main__":
    retriever = Retriever("models/retrieval")
    query = "Who is the Dean of the College of Computing, Data Science, and Society?"
    print(f"Query: {query}")
    results = retriever.retrieve(query, k=5)
    for i, res in enumerate(results):
        print(f"{i+1}. File: {res['metadata']['file']}")
        print(f"Content: {res['content'][:200]}...")
        print("-" * 20)
