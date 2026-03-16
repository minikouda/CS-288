import os
import json
import re
import hashlib
import asyncio
import logging
import argparse
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

RAW_DIR = "data/raw"
CLEANED_DIR = "data/cleaned"

# Regex for common boilerplate
BOILERPLATE_PATTERNS = [
    r"View Open Faculty Positions$",
    r"1 2 Next Page \u00bb",
    r"Faculty Archives - EECS at Berkeley Faculty",
    r"Subscribe to our newsletter",
    r"Follow us on Twitter",
    r"EECS at UC Berkeley",
    r"Contact us"
]

def clean_content(text):
    """Surgical cleaning of content to remove boilerplate but preserve facts."""
    if not text:
        return ""
    
    # 1. Strip redundant whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 2. Remove known boilerplate patterns
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # 3. Basic unicode normalization
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    return text.strip()

def process_file_sync(file_path):
    """Synchronous file reading and processing."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            item = json.load(f)
            
            if "content" not in item:
                return None
            
            clean_text = clean_content(item["content"])
            
            # Skip if too short after cleaning
            if len(clean_text) < 100:
                return None
            
            item["content"] = clean_text
            
            # Generate a unique hash for deduplication
            content_hash = hashlib.md5(clean_text.encode()).hexdigest()
            item["content_hash"] = content_hash
            
            return item
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return None

def process_line_sync(line):
    """Synchronous line processing for JSONL."""
    try:
        item = json.loads(line)
        if "content" not in item:
            return None
        
        clean_text = clean_content(item["content"])
        if len(clean_text) < 100:
            return None
            
        item["content"] = clean_text
        content_hash = hashlib.md5(clean_text.encode()).hexdigest()
        item["content_hash"] = content_hash
        return item
    except Exception:
        return None

async def run_async_cleaner(input_source, output_file, is_directory=True):
    """Main entry point to clean and deduplicate data using multiprocessing."""
    logging.info(f"Starting async data cleaning from {input_source}...")
    
    processed_data = []
    seen_hashes = set()
    
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor() as executor:
        if is_directory:
            # Process individual JSON files
            json_files = [os.path.join(input_source, f) for f in os.listdir(input_source) if f.endswith('.json')]
            logging.info(f"Found {len(json_files)} JSON files to process.")
            
            # Run in process pool
            tasks = [loop.run_in_executor(executor, process_file_sync, f) for f in json_files]
            results = await asyncio.gather(*tasks)
            
            for item in results:
                if item and item["content_hash"] not in seen_hashes:
                    seen_hashes.add(item["content_hash"])
                    processed_data.append(item)
        else:
            # Process JSONL file
            with open(input_source, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                logging.info(f"Read {len(lines)} lines from JSONL.")
                
                tasks = [loop.run_in_executor(executor, process_line_sync, line) for line in lines]
                results = await asyncio.gather(*tasks)
                
                for item in results:
                    if item and item["content_hash"] not in seen_hashes:
                        seen_hashes.add(item["content_hash"])
                        processed_data.append(item)

    # Write output JSONL
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item) + '\n')
            
    logging.info(f"Cleaning Complete! Saved {len(processed_data)} clean pages to {output_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and deduplicate scraped data.")
    parser.add_argument("--input", default=RAW_DIR, help="Path to raw JSON files or JSONL")
    parser.add_argument("--output", default=None, help="Output JSONL path")
    
    args = parser.parse_args()
    
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(CLEANED_DIR, f"corpus_clean_{timestamp}.jsonl")
    
    is_dir = os.path.isdir(args.input)
    asyncio.run(run_async_cleaner(args.input, args.output, is_directory=is_dir))
