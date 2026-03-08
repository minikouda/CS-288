import os
import json
import re
from collections import Counter
import pandas as pd

RAW_DIR = "data/processed"

def run_eda():
    data = []
    files = [f for f in os.listdir(RAW_DIR) if f.endswith('.json')]
    
    if not files:
        print(f"No files found in {RAW_DIR}")
        return

    print(f"Analyzing {len(files)} files...")
    
    for f in files:
        path = os.path.join(RAW_DIR, f)
        try:
            with open(path, 'r', encoding='utf-8') as file:
                item = json.load(file)
                # Filter out minimal content
                content = item.get('content', '')
                if not content:
                    continue
                item['filename'] = f
                item['char_count'] = len(content)
                item['word_count'] = len(content.split())
                data.append(item)
        except Exception as e:
            # print(f"Error loading {f}: {e}")
            continue
    
    if not data:
        print("No valid data loaded.")
        return

    df = pd.DataFrame(data)
    
    # 1. Duplicate URL Analysis
    url_counts = df['url'].value_counts()
    dupe_urls = url_counts[url_counts > 1]
    print(f"\n--- URL Duplication ---")
    print(f"Total Unique URLs: {len(url_counts)}")
    print(f"URLs with multiple files: {len(dupe_urls)}")
    if not dupe_urls.empty:
        print(f"Sample duplicates:\n{dupe_urls.head()}")
    
    # 2. Content Length Analysis
    print(f"\n--- Content Length Stats ---")
    print(df[['char_count', 'word_count']].describe())
    
    # 3. Boilerplate / Suffix Analysis
    # Get last 100 characters to identify common footer patterns
    suffixes = [c[-100:].strip() for c in df['content'] if len(c) > 100]
    common_suffixes = Counter(suffixes).most_common(15)
    print(f"\n--- Top 15 Common Suffixes (Boilerplate candidates) ---")
    for s, count in common_suffixes:
        print(f"[{count} times]: {repr(s)}")

    # 4. Prefix Analysis
    prefixes = [c[:100].strip() for c in df['content'] if len(c) > 100]
    common_prefixes = Counter(prefixes).most_common(10)
    print(f"\n--- Top 10 Common Prefixes (Header boilerplate candidates) ---")
    for s, count in common_prefixes:
        print(f"[{count} times]: {repr(s)}")

    # 5. Short Page Analysis
    short_pages = df[df['word_count'] < 30]
    print(f"\n--- Short Pages (< 30 words) ---")
    print(f"Count: {len(short_pages)}")
    if not short_pages.empty:
        print(short_pages[['url', 'title']].head(10))

if __name__ == "__main__":
    run_eda()
