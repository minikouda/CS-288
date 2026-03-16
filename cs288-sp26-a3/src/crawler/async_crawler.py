import asyncio
import aiohttp
from bs4 import BeautifulSoup
import re
import os
import json
from urllib.parse import urljoin, urlparse, urldefrag
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
BASE_URL = "https://eecs.berkeley.edu"
# Regex strictly from the assignment guidelines
ALLOWED_REGEX = re.compile(r"^https?://(?:www\d*\.)?eecs\.berkeley\.edu(?:/[^\s]*)?$")
# Added common dynamic trap extensions like .ics or calendar views
IGNORE_EXTENSIONS = {'.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.gz', '.tar', '.mp4', '.mov', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.ics'}
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "corpus_raw.jsonl")

# Concurrency limits
MAX_CONCURRENT_REQUESTS = 20

def setup_directory():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # Clear out the file if it exists to start fresh
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

def normalize_url(url):
    """Remove fragments and trailing slashes to prevent duplicate crawling."""
    url, _ = urldefrag(url)
    if url.endswith('/') and len(url) > len(BASE_URL) + 1:
        url = url[:-1]
    return url

def is_valid_url(url):
    """Check if the URL matches the required regex and is not an ignored file."""
    if not ALLOWED_REGEX.match(url):
        return False
    
    parsed = urlparse(url)
    path = parsed.path.lower()
    
    # Check extensions
    if any(path.endswith(ext) for ext in IGNORE_EXTENSIONS):
        return False
        
    # Trap prevention: Avoid obvious infinite calendar/sorting parameters
    query = parsed.query.lower()
    if 'month=' in query or 'year=' in query or 'sort=' in query or 'event=' in query:
        return False
        
    return True

async def fetch_and_parse(session, url, semaphore):
    """Fetch the HTML and parse out the clean text and new links."""
    async with semaphore:
        try:
            # Polite delay to prevent overwhelming the server
            await asyncio.sleep(0.1)
            async with session.get(url, timeout=15, allow_redirects=True) as response:
                if response.status != 200:
                    if response.status == 403:
                        logging.warning(f"403 Forbidden on {url}. Adjusting politeness.")
                        await asyncio.sleep(5) # Back off on 403
                    return None, []
                
                content_type = response.headers.get("Content-Type", "").lower()
                if "text/html" not in content_type:
                    return None, []

                html = await response.text()
                soup = BeautifulSoup(html, 'lxml')
                
                title = soup.title.string.strip() if soup.title and soup.title.string else ""
                
                # Extract new links to crawl
                new_links = []
                for link in soup.find_all(['a', 'iframe']):
                    raw_href = link.get('href') or link.get('src')
                    if not raw_href:
                        continue
                        
                    full_url = normalize_url(urljoin(url, raw_href))
                    if is_valid_url(full_url):
                        new_links.append(full_url)

                # Clean the text
                for element in soup(["script", "style", "nav", "footer", "header"]):
                    element.decompose()
                text = soup.get_text(separator=' ', strip=True)
                clean_text = re.sub(r'\s+', ' ', text).strip()
                
                return {"url": url, "title": title, "content": clean_text}, new_links
                
        except asyncio.TimeoutError:
            logging.debug(f"Timeout fetching {url}")
            return None, []
        except Exception as e:
            logging.debug(f"Error fetching {url}: {str(e)}")
            return None, []

async def crawl_worker(name, queue, session, semaphore, visited, file_lock, outfile, stats):
    """Worker task that constantly pulls URLs from the queue and processes them."""
    while True:
        try:
            url = await queue.get()
            
            # Stop condition based on our target limit
            if stats['saved'] >= stats['max_pages']:
                queue.task_done()
                continue

            page_data, new_links = await fetch_and_parse(session, url, semaphore)
            
            if page_data and len(page_data['content']) > 50: # Skip empty/meaningless pages
                async with file_lock:
                    outfile.write(json.dumps(page_data) + "\n")
                    outfile.flush()
                stats['saved'] += 1
                if stats['saved'] % 100 == 0:
                    logging.info(f"Progress: {stats['saved']}/{stats['max_pages']} pages saved. Queue size: {queue.qsize()}")

            # Add valid new links to the queue
            for link in new_links:
                if link not in visited and stats['saved'] + queue.qsize() < stats['max_pages'] * 1.5: # Don't overfill queue
                    visited.add(link)
                    queue.put_nowait(link)

            queue.task_done()
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logging.error(f"Worker {name} encountered error: {e}")
            queue.task_done()

async def main(max_pages=20000):
    setup_directory()
    start_time = time.time()
    
    visited = set([BASE_URL])
    queue = asyncio.Queue()
    queue.put_nowait(BASE_URL)
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    file_lock = asyncio.Lock()
    stats = {'saved': 0, 'max_pages': max_pages}
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }
    
    logging.info(f"Starting async scrape to target {max_pages} pages...")
    
    # Open the JSONL file in append mode
    with open(OUTPUT_FILE, "a", encoding="utf-8") as outfile:
        async with aiohttp.ClientSession(headers=headers) as session:
            # Spin up workers
            workers = [
                asyncio.create_task(crawl_worker(f"w-{i}", queue, session, semaphore, visited, file_lock, outfile, stats))
                for i in range(MAX_CONCURRENT_REQUESTS)
            ]
            
            # # Wait until the queue is fully processed or we hit our page limit
            # while not queue.empty() and stats['saved'] < max_pages:
            #     await asyncio.sleep(2)
            # Create a task to monitor when the queue is fully processed
            queue_task = asyncio.create_task(queue.join())
            
            # Wait until either the queue is completely done OR we hit our max_pages limit
            while stats['saved'] < max_pages:
                if queue_task.done():
                    break
                await asyncio.sleep(1)
                
            # Cancel workers once done
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            
    elapsed = time.time() - start_time
    logging.info(f"Scraping complete! Saved {stats['saved']} pages in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    # Windows-specific fix for asyncio loop
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(max_pages=20000))