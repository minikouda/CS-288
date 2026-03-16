import os
import json
import re
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

RAW_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "raw", "corpus_raw.jsonl")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "processed")
PROCESSED_FILE = os.path.join(PROCESSED_DIR, "corpus_clean.jsonl")

def setup_directories():
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
    # Clear old processed file if it exists
    if os.path.exists(PROCESSED_FILE):
        os.remove(PROCESSED_FILE)

def clean_text(text):
    """Normalize and strip boilerplate."""
    # 1. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 2. Remove known boilerplate suffixes (Iterative cleaning)
    suffixes_to_remove = [
        "View Open Faculty Positions",
        "Staff Positions Available",
        "Faculty Positions Available",
        "Joseph Gier Memorial Project",
        "Our Leadership",
        "Student Affairs Staff",
        "Course Support Staff",
        "Human Resources Staff",
        "Apple NSI Fellowship and Scholarship Recipients",
        "Apple NSI Course Flow Map",
        "Pursue Your Research Interests",
        "Community College Day",
        "Attend Community College Day (Fall Only)",
        "Attend Cal Day (Spring Only) Cal Day",
        "Archive of Special Events",
    ]

    changed = True
    while changed:
        old_len = len(text)
        for suffix in suffixes_to_remove:
            if text.endswith(suffix):
                text = text[:-len(suffix)].strip()

        # Remove pagination patterns
        text = re.sub(r'\d+ \d+ (?:3|4|5|6|7|8|9|…) .* Next Page »$', '', text).strip()
        text = re.sub(r'1 2 Next Page »$', '', text).strip()

        changed = len(text) < old_len

    # 3. Unicode normalization
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u2013', '-').replace('\u2014', '--')
    text = text.replace('\u00bb', '>>')

    return text

def run_cleaner():
    setup_directories()
    
    if not os.path.exists(RAW_FILE):
        logging.error(f"Raw file not found at {RAW_FILE}. Run the scraper first!")
        return

    seen_urls = set()
    seen_hashes = set()
    processed_count = 0
    skipped_count = 0

    logging.info("Starting data cleaning and deduplication...")

    with open(RAW_FILE, 'r', encoding='utf-8') as infile, \
         open(PROCESSED_FILE, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            try:
                item = json.loads(line)
                url = item.get('url', '')
                raw_content = item.get('content', '')

                # Skip if we've already seen this exact URL
                if url in seen_urls:
                    skipped_count += 1
                    continue
                
                # Clean the content
                cleaned_content = clean_text(raw_content)
                
                # Skip if too short after cleaning (less than 15 words)
                if len(cleaned_content.split()) < 15:
                    skipped_count += 1
                    continue

                # Deduplicate based on content hash to catch identical pages at different URLs
                content_hash = hashlib.md5(cleaned_content.encode('utf-8')).hexdigest()
                if content_hash in seen_hashes:
                    skipped_count += 1
                    continue

                # Record as seen
                seen_urls.add(url)
                seen_hashes.add(content_hash)

                # Save the cleaned item
                clean_item = {
                    "url": url,
                    "title": item.get('title', ''),
                    "content": cleaned_content
                }
                outfile.write(json.dumps(clean_item) + "\n")
                
                processed_count += 1
                if processed_count % 1000 == 0:
                    logging.info(f"Cleaned {processed_count} unique pages...")

            except json.JSONDecodeError:
                continue

    logging.info(f"Cleaning Complete! Saved {processed_count} clean pages. Skipped {skipped_count} duplicates/short pages.")

if __name__ == "__main__":
    run_cleaner()