import json
import glob
import os
import argparse
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


def iter_documents(data_source):
    """Yield (doc, source_name) from a directory of JSON files or a JSONL file."""
    if os.path.isfile(data_source):
        if data_source.endswith(".jsonl"):
            source_name = os.path.basename(data_source)
            with open(data_source, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line), f"{source_name}:L{line_no}"
                    except json.JSONDecodeError:
                        continue
        elif data_source.endswith(".json"):
            with open(data_source, 'r', encoding='utf-8') as f:
                yield json.load(f), os.path.basename(data_source)
        return

    json_files = sorted(glob.glob(os.path.join(data_source, "*.json")))
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            yield json.load(f), os.path.basename(file_path)

def build_index(data_dir, output_dir, model_name="BAAI/bge-base-en-v1.5"):
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
    
    print("Processing documents with Recursive Chunking...")
    for data, source_name in tqdm(list(iter_documents(data_dir))):
        url = data.get("url", "")
        title = data.get("title", "")

        # Use clean_markdown if available, else standard content
        content = data.get("clean_markdown", data.get("content", ""))
        summary = data.get("summary", "")

        # 1. Page Summary Chunk (High level context)
        if summary:
            summary_text = f"Summary: {summary}\nTitle: {title}\nURL: {url}"
            all_chunks.append(summary_text)
            chunk_metadata.append({"url": url, "title": title, "file": source_name, "type": "summary"})

        # 2. Detailed Chunks
        full_text = f"Title: {title}\nURL: {url}\nContent: {content}"
        chunks = recursive_chunk_text(full_text, chunk_size=1000, overlap=200)

        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_metadata.append({
                "url": url,
                "title": title,
                "file": source_name,
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
    parser = argparse.ArgumentParser(description="Build hybrid retrieval index from JSON pages or JSONL corpus.")
    parser.add_argument("--data", default="data/processed", help="Input data source (directory of .json or a .jsonl file)")
    parser.add_argument("--output", default="models/retrieval", help="Output index directory")
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5", help="SentenceTransformer embedding model")
    args = parser.parse_args()

    build_index(args.data, args.output, model_name=args.model)
