"""
Merge two JSONL corpora and deduplicate by URL, then by content hash.
Priority: the first file listed takes precedence on URL conflicts.

Usage:
    python -m src.processing.merge_corpora \
        --inputs data/eecs_text_bs_rewritten_mapped.jsonl data/cleaned/corpus_clean_v5.jsonl \
        --output data/cleaned/corpus_merged.jsonl

Then build the index:
    python -m src.retrieval.build_index \
        --data data/cleaned/corpus_merged.jsonl \
        --output models/retrieval_total
"""

import json
import hashlib
import argparse
import os


def content_hash(text):
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def merge_corpora(input_paths, output_path):
    seen_urls = {}      # url -> doc (first occurrence wins)
    seen_hashes = set() # content-level dedup after URL dedup

    for path in input_paths:
        print(f"Reading {path}...")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue

                url = doc.get("url", "").strip()
                content = doc.get("content", doc.get("text", "")).strip()

                if not content:
                    continue

                # URL dedup: first file wins
                if url and url in seen_urls:
                    continue

                # Content hash dedup
                h = content_hash(content)
                if h in seen_hashes:
                    continue

                # Normalise to common schema
                seen_urls[url] = {
                    "url": url,
                    "title": doc.get("title", ""),
                    "content": content,
                    "content_hash": h,
                }
                seen_hashes.add(h)

    docs = list(seen_urls.values())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")

    print(f"Merged {len(docs)} unique documents -> {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and deduplicate JSONL corpora.")
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="Input JSONL files in priority order (first file wins on URL conflicts)"
    )
    parser.add_argument(
        "--output", default="data/cleaned/corpus_merged.jsonl",
        help="Output JSONL path"
    )
    args = parser.parse_args()
    merge_corpora(args.inputs, args.output)
