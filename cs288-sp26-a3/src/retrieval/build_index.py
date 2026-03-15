import json
import glob
import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from rank_bm25 import BM25Okapi

def chunk_text(text, chunk_size=500, overlap=100):
    """Simple sliding window chunking."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def tokenize(text):
    return text.lower().split()

def build_index(data_dir, output_dir, model_name="thenlper/gte-base"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load embedding model
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    # Load and chunk documents
    all_chunks = []
    chunk_metadata = []
    
    print("Processing files...")
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    for file_path in tqdm(json_files):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            url = data.get("url", "")
            title = data.get("title", "")
            content = data.get("content", "")
            
            full_text = f"Title: {title}\nURL: {url}\nContent: {content}"
            # Use chunks of 2000 chars with 200 overlap to handle long pages
            # but keep most pages in 1-2 chunks
            chunks = chunk_text(full_text, chunk_size=2000, overlap=200)
            
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "url": url,
                    "title": title,
                    "file": os.path.basename(file_path)
                })
    
    print(f"Total chunks: {len(all_chunks)}")
    
    # Encode chunks for Dense Retrieval
    print("Encoding chunks for Dense Retrieval...")
    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
    
    # Build FAISS index
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Build BM25 index
    print("Building BM25 index...")
    tokenized_corpus = [tokenize(chunk) for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save index and metadata
    faiss.write_index(index, os.path.join(output_dir, "index.faiss"))
    with open(os.path.join(output_dir, "bm25.pkl"), 'wb') as f:
        pickle.dump(bm25, f)
    with open(os.path.join(output_dir, "chunks.json"), 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f)
    with open(os.path.join(output_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(chunk_metadata, f)
    
    print("Index and metadata saved successfully.")

if __name__ == "__main__":
    build_index("data/processed", "models/retrieval")

if __name__ == "__main__":
    build_index("data/processed", "models/retrieval")
