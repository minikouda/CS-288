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

def recursive_chunk_text(text, chunk_size=800, overlap=150):
    """Splits text recursively on double newlines, single newlines, and spaces."""
    if len(text) <= chunk_size:
        return [text]
    
    separators = ["\n\n", "\n", " ", ""]
    final_chunks = []
    
    # Simple recursive splitting logic
    def split_recursive(txt):
        if len(txt) <= chunk_size:
            return [txt]
        
        for sep in separators:
            if sep == "":
                # Fallback to hard cut
                return [txt[i:i+chunk_size] for i in range(0, len(txt), chunk_size - overlap)]
            
            if sep in txt:
                parts = txt.split(sep)
                temp_chunk = ""
                chunks = []
                for p in parts:
                    if len(temp_chunk) + len(p) + len(sep) <= chunk_size:
                        temp_chunk += p + sep
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = p + sep
                if temp_chunk:
                    chunks.append(temp_chunk.strip())
                
                # If we made progress, return
                if len(chunks) > 1:
                    return chunks
        return [txt] # Should not reach here

    raw_chunks = split_recursive(text)
    
    # Handle overlap for the chunks
    # (Simple version: just ensure we don't exceed size)
    return raw_chunks

def build_index(data_dir, output_dir, model_name="all-MiniLM-L12-v2"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if augmented data exists, otherwise fallback to processed
    aug_dir = "data/augmented"
    if os.path.exists(aug_dir) and len(os.listdir(aug_dir)) > 0:
        print(f"Using AUGMENTED data from {aug_dir}")
        data_dir = aug_dir
    else:
        print(f"Using standard data from {data_dir}")
    
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    all_chunks = []
    chunk_metadata = []
    
    print("Processing files with Recursive Chunking...")
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    for file_path in tqdm(json_files):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            url = data.get("url", "")
            title = data.get("title", "")
            
            # Use clean_markdown if available, else standard content
            content = data.get("clean_markdown", data.get("content", ""))
            summary = data.get("summary", "")
            
            # 1. Page Summary Chunk (High level context)
            if summary:
                summary_text = f"Summary: {summary}\nTitle: {title}\nURL: {url}"
                all_chunks.append(summary_text)
                chunk_metadata.append({"url": url, "title": title, "file": os.path.basename(file_path), "type": "summary"})

            # 2. Detailed Chunks
            full_text = f"Title: {title}\nURL: {url}\nContent: {content}"
            chunks = recursive_chunk_text(full_text, chunk_size=1000, overlap=200)
            
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "url": url,
                    "title": title,
                    "file": os.path.basename(file_path),
                    "type": "detail"
                })
    
    print(f"Total chunks: {len(all_chunks)}")
    
    print("Encoding chunks...")
    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
    
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    print("Building BM25 index...")
    tokenized_corpus = [tokenize(chunk) for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save
    faiss.write_index(index, os.path.join(output_dir, "index.faiss"))
    with open(os.path.join(output_dir, "bm25.pkl"), 'wb') as f:
        pickle.dump(bm25, f)
    with open(os.path.join(output_dir, "chunks.json"), 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f)
    with open(os.path.join(output_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(chunk_metadata, f)
    
    print("Index saved successfully.")

if __name__ == "__main__":
    build_index("data/processed", "models/retrieval")

if __name__ == "__main__":
    build_index("data/processed", "models/retrieval")
