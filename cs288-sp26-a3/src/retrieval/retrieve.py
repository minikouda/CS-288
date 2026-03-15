import faiss
import json
import os
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

def tokenize(text):
    return text.lower().split()

class Retriever:
    def __init__(self, index_dir, model_name="thenlper/gte-base"):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        with open(os.path.join(index_dir, "bm25.pkl"), 'rb') as f:
            self.bm25 = pickle.load(f)
        with open(os.path.join(index_dir, "chunks.json"), 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        with open(os.path.join(index_dir, "metadata.json"), 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

    def retrieve_dense(self, query, k=5):
        # gte-base works well with direct query encoding
        query_vector = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vector, k)
        results = []
        for i in range(k):
            idx = indices[0][i]
            if idx != -1:
                results.append(idx)
        return results

    def retrieve_bm25(self, query, k=5):
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[-k:][::-1]
        return top_n.tolist()

    def retrieve(self, query, k=10):
        # Hybrid: use Reciprocal Rank Fusion (RRF)
        dense_indices = self.retrieve_dense(query, k=20) # Get more candidates for fusion
        bm25_indices = self.retrieve_bm25(query, k=20)
        
        # RRF scoring
        rrf_scores = {}
        # Higher k in RRF (60) is standard to avoid over-weighting top ranks
        for rank, idx in enumerate(dense_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (60 + rank)
            
        for rank, idx in enumerate(bm25_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (60 + rank)
            
        # Sort by RRF score
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        final_indices = sorted_indices[:k]
        
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
    for res in results:
        print(f"File: {res['metadata']['file']}")
        print(f"Content: {res['content'][:300]}...")
        print("-" * 20)
