import requests
from bs4 import BeautifulSoup
import re
import os
import json
from urllib.parse import urljoin, urlparse
import time
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
BASE_URL = "https://eecs.berkeley.edu"
ALLOWED_REGEX = re.compile(r"^https?://(?:www\d*\.)?eecs\.berkeley\.edu(?:/[^\s]*)?$")
IGNORE_EXTENSIONS = {'.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.gz', '.tar', '.mp4', '.mov', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx'}
# Note: When running from src/crawler/, we need to go up one level to reach data/raw
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "raw")

def setup_directory():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def is_valid_url(url):
    """Check if the URL matches the required regex and is not a binary/ignored file."""
    if not ALLOWED_REGEX.match(url):
        return False
    
    path = urlparse(url).path.lower()
    if any(path.endswith(ext) for ext in IGNORE_EXTENSIONS):
        return False
        
    return True

def get_page_content(session, url, depth=0, max_depth=2):
    """Recursively fetch content, following iframes."""
    if depth > max_depth:
        return ""

    try:
        # Respectful delay
        time.sleep(1.0 if depth > 0 else 2.0)
        
        response = session.get(url, timeout=15)
        if response.status_code != 200:
            return ""
        
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type:
            return ""

        soup = BeautifulSoup(response.text, 'lxml')
        
        # Remove unwanted tags only from the top-level call (or handle carefully)
        # We want to keep iframe sources before we decompose them
        iframes = soup.find_all('iframe', src=True)
        iframe_contents = []
        for iframe in iframes:
            iframe_url = urljoin(url, iframe['src'])
            if is_valid_url(iframe_url):
                logging.info(f"Following iframe: {iframe_url} (depth {depth+1})")
                iframe_contents.append(get_page_content(session, iframe_url, depth + 1, max_depth))
        
        # Now clean the main soup
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
            
        text = soup.get_text(separator=' ', strip=True)
        
        # Merge with iframe contents
        full_text = text + " " + " ".join(iframe_contents)
        return re.sub(r'\s+', ' ', full_text).strip()

    except Exception as e:
        logging.error(f"Error in get_page_content for {url}: {e}")
        return ""

def scrape_eecs(max_pages=3000):
    setup_directory()
    
    visited = set()
    queue = deque([BASE_URL])
    
    pages_saved = 0
    
    # Create a session to handle cookies and headers consistently
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com/'
    })
    
    logging.info(f"Starting scrape from {BASE_URL} (max_pages={max_pages})...")
    
    while queue and pages_saved < max_pages:
        url = queue.popleft()
        
        # Remove fragment identifier if any
        url = url.split('#')[0]
        
        if url in visited:
            continue
            
        visited.add(url)
        
        if not is_valid_url(url):
            continue
            
        try:
            # Respectful crawling with a larger delay to avoid 403
            time.sleep(1.0) 
            
            response = session.get(url, timeout=15)
            
            if response.status_code != 200:
                logging.warning(f"Failed to fetch {url}: Status {response.status_code}")
                # If we get a 403, we might want to wait longer or stop
                if response.status_code == 403:
                    logging.error("Access Forbidden (403). Waiting 60 seconds...")
                    time.sleep(60)
                continue
                
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type:
                continue

            soup = BeautifulSoup(response.text, 'lxml')
            title = soup.title.string if soup.title else ""
            
            # Use the recursive function to get full content including iframes
            text_content = get_page_content(session, url)
            
            if not text_content:
                continue

            # Save to corpus
            page_data = {
                "url": url,
                "title": title,
                "content": text_content
            }
            
            # Generate a safe filename based on the URL
            parsed_url = urlparse(url)
            safe_name = re.sub(r'[^a-zA-Z0-9]', '_', parsed_url.path)
            safe_name = safe_name.strip('_')
            if not safe_name:
                safe_name = "index"
            
            filename = f"{safe_name}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            # Handle duplicates if paths map to same safe name
            counter = 1
            while os.path.exists(filepath):
                filename = f"{safe_name}_{counter}.json"
                filepath = os.path.join(OUTPUT_DIR, filename)
                counter += 1
                
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(page_data, f, indent=2)
                
            pages_saved += 1
            logging.info(f"[{pages_saved}/{max_pages}] Saved: {url}")
            
            # Find more links
            for link in soup.find_all('a', href=True):
                new_url = urljoin(url, link['href']).split('#')[0]
                if is_valid_url(new_url) and new_url not in visited:
                    queue.append(new_url)
                    
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error scraping {url}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error scraping {url}: {e}")
            
    logging.info(f"Scraping complete. Total pages saved: {pages_saved}")

if __name__ == "__main__":
    scrape_eecs(max_pages=3000)
