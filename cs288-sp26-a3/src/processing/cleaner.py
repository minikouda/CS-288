import os
import json
import re
import shutil

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

# Regex for common boilerplate
BOILERPLATE_PATTERNS = [
    r"View Open Faculty Positions$",
    r"1 2 Next Page \u00bb",
    r"Home - EECS at Berkeley.*Welcome!",
    r"EECS at Berkeley EECS at Berkeley",
]

def setup_directories():
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)
    os.makedirs(PROCESSED_DIR)
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

    # Repeat suffix removal to handle nested boilerplate
    changed = True
    while changed:
        old_len = len(text)
        for suffix in suffixes_to_remove:
            if text.endswith(suffix):
                text = text[:-len(suffix)].strip()

        # Remove pagination patterns like "1 2 3 ... 11 Next Page »"
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
    
    files = [f for f in os.listdir(RAW_DIR) if f.endswith('.json')]
    
    # Deduplication logic: map URL to the best filename
    url_to_file = {}
    
    for f in files:
        path = os.path.join(RAW_DIR, f)
        try:
            with open(path, 'r', encoding='utf-8') as file:
                item = json.load(file)
                url = item.get('url')
                content = item.get('content', '')
                
                if not url: continue
                
                # Keep the one with more content if duplicate URL
                if url not in url_to_file or len(content) > url_to_file[url][1]:
                    url_to_file[url] = (f, len(content), item)
        except:
            continue
            
    print(f"Deduplicated {len(files)} files down to {len(url_to_file)} unique URLs.")
    
    processed_count = 0
    for url, (orig_file, size, item) in url_to_file.items():
        content = item.get('content', '')
        
        # Clean the content
        cleaned_content = clean_text(content)
        
        # Filter: Skip if too short after cleaning
        if len(cleaned_content.split()) < 15:
            continue
            
        item['content'] = cleaned_content
        item['original_file'] = orig_file
        
        # Save to processed directory
        dest_filename = f"page_{processed_count:04d}.json"
        with open(os.path.join(PROCESSED_DIR, dest_filename), 'w', encoding='utf-8') as f:
            json.dump(item, f, indent=2)
            
        processed_count += 1
        
    print(f"Cleaning complete. Saved {processed_count} unique, cleaned pages to {PROCESSED_DIR}.")

if __name__ == "__main__":
    run_cleaner()
