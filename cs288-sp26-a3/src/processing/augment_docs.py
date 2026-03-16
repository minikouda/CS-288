import json
import glob
import os
import argparse
from tqdm import tqdm
from src.llm import call_llm

def augment_document(content, title, url):
    system_prompt = (
        "You are an expert data cleaner. Your task is to process raw text from a UC Berkeley EECS webpage "
        "and return a JSON object with two fields:\n"
        "1. 'clean_markdown': The content rewritten in clean, structured Markdown (preserving tables and lists).\n"
        "2. 'summary': A 2-sentence summary of the page's core purpose."
    )
    
    prompt = f"Title: {title}\nURL: {url}\n\nRaw Content:\n{content[:4000]}" # Limit to fit context
    
    try:
        response = call_llm(
            query=prompt,
            system_prompt=system_prompt,
            model="meta-llama/llama-3.1-8b-instruct",
            max_tokens=1000,
            temperature=0.0
        )
        # Extract JSON from response (handling potential markdown blocks)
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
            
        return json.loads(response.strip())
    except Exception as e:
        print(f"Error augmenting doc: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10, help="Number of files to process (for testing)")
    args = parser.parse_args()

    input_dir = "data/processed"
    output_dir = "data/augmented"
    os.makedirs(output_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(input_dir, "*.json"))
    files.sort()
    
    if args.limit:
        files = files[:args.limit]
        
    print(f"Augmenting {len(files)} documents...")
    
    for file_path in tqdm(files):
        out_path = os.path.join(output_dir, os.path.basename(file_path))
        
        # Checkpoint: skip if already processed
        if os.path.exists(out_path):
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        augmented = augment_document(data['content'], data['title'], data['url'])
        
        if augmented:
            data['clean_markdown'] = augmented.get('clean_markdown', '')
            data['summary'] = augmented.get('summary', '')
            
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

if __name__ == "__main__":
    main()
