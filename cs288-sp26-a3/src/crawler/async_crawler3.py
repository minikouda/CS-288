import asyncio
import aiohttp
from bs4 import BeautifulSoup
import re
import os
import json
import random
from urllib.parse import urljoin, urlparse, urldefrag
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

BASE_URL = "https://eecs.berkeley.edu"
ALLOWED_REGEX = re.compile(r"^https?://(?:www\d*\.)?eecs\.berkeley\.edu(?:/[^\s]*)?$")
IGNORE_EXTENSIONS = {'.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.gz', '.tar', 
                     '.mp4', '.mov', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.ics'}

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "corpus_raw5.jsonl")

MAX_CONCURRENT_REQUESTS = 5
MAX_RETRIES = 5
COOLDOWN_ON_403 = 60

def setup_directory():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def load_state():
    visited = set()
    saved_count = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    visited.add(data['url'])
                    saved_count += 1
                except:
                    pass
    return visited, saved_count

def normalize_url(url):
    url, _ = urldefrag(url)
    if url.endswith('/') and len(url) > len(BASE_URL) + 1:
        url = url[:-1]
    return url

def is_valid_url(url):
    if not ALLOWED_REGEX.match(url):
        return False
    parsed = urlparse(url)
    if any(parsed.path.lower().endswith(ext) for ext in IGNORE_EXTENSIONS):
        return False
    query = parsed.query.lower()
    if 'month=' in query or 'year=' in query or 'sort=' in query or 'event=' in query:
        return False
    return True

async def fetch_and_parse(session, url, semaphore):
    async with semaphore:
        await asyncio.sleep(random.uniform(1.0, 3.0))  # human-like jitter
        
        html = None
        
        # Retry logic and fetching combined safely
        for attempt in range(MAX_RETRIES):
            try:
                timeout = aiohttp.ClientTimeout(total=60)
                async with session.get(url, timeout=timeout, allow_redirects=True) as response:
                    status = response.status
                    
                    if status == 403 or status == 429:
                        logging.warning(f"Rate Limit/WAF triggered on {url} (Status {status}). Cooling down {COOLDOWN_ON_403}s...")
                        await asyncio.sleep(COOLDOWN_ON_403)
                        return None, []
                    elif status != 200:
                        logging.info(f"Skipping {url} - Status Code: {status}")
                        return None, []

                    content_type = response.headers.get("Content-Type", "").lower()
                    if "text/html" not in content_type:
                        return None, []

                    # CRITICAL FIX: Read the text INSIDE the connection block!
                    html = await response.text()
                    break # Success! Break out of the retry loop.

            except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionResetError) as e:
                logging.warning(f"Attempt {attempt+1}/{MAX_RETRIES} failed for {url}: {e}")
                await asyncio.sleep(2 ** attempt + random.uniform(0, 1))
                
                # HTTP Fallback logic if HTTPS completely fails
                if attempt == MAX_RETRIES - 1 and url.startswith("https://"):
                    http_url = url.replace("https://", "http://", 1)
                    logging.info(f"Falling back to HTTP for {url}")
                    try:
                        async with session.get(http_url, timeout=timeout, allow_redirects=True) as fallback_resp:
                            if fallback_resp.status == 200 and "text/html" in fallback_resp.headers.get("Content-Type", "").lower():
                                html = await fallback_resp.text()
                    except Exception as e2:
                        logging.error(f"HTTP fallback failed for {url}: {e2}")

        # If all retries failed and we have no HTML, bail out early
        if not html:
            return None, []

        # Now that we have the HTML safely saved in memory, we can parse it
        try:
            soup = BeautifulSoup(html, 'html.parser')
            title = soup.title.string.strip() if soup.title and soup.title.string else ""

            new_links = []
            for link in soup.find_all(['a', 'iframe']):
                raw_href = link.get('href') or link.get('src')
                if raw_href:
                    full_url = normalize_url(urljoin(url, raw_href))
                    if is_valid_url(full_url):
                        new_links.append(full_url)

            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            text = soup.get_text(separator=' ', strip=True)
            clean_text = re.sub(r'\s+', ' ', text).strip()
            return {"url": url, "title": title, "content": clean_text}, new_links

        except Exception as e:
            logging.error(f"Error parsing {url}: {e}")
            return None, []

async def crawl_worker(queue, session, semaphore, visited, file_lock, outfile, stats):
    while True:
        try:
            url = await queue.get()
            if stats['saved'] >= stats['max_pages']:
                queue.task_done()
                continue

            page_data, new_links = await fetch_and_parse(session, url, semaphore)
            if page_data and len(page_data['content']) > 50:
                async with file_lock:
                    outfile.write(json.dumps(page_data) + "\n")
                    outfile.flush()
                stats['saved'] += 1
                if stats['saved'] % 5 == 0:
                    logging.info(f"🏆 Progress: {stats['saved']}/{stats['max_pages']} pages saved. Queue size: {queue.qsize()}")

            for link in new_links:
                if link not in visited:
                    visited.add(link)
                    queue.put_nowait(link)
            queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logging.error(f"Worker error: {e}")
            queue.task_done()

async def main(max_pages=5000):
    setup_directory()
    visited, saved_count = load_state()
    logging.info(f"Starting marathon crawl. Current saved pages: {saved_count}")

    seed_urls = [
        "https://www2.eecs.berkeley.edu/Pubs/TechRpts/",
        "https://www2.eecs.berkeley.edu/Faculty/Lists/faculty.html",
        "https://www2.eecs.berkeley.edu/Research/Areas/",
        "https://www2.eecs.berkeley.edu/Courses/"
    ]

    queue = asyncio.Queue()
    for url in seed_urls:
        if url not in visited:
            visited.add(url)
            queue.put_nowait(url)
            
    if queue.empty():
        logging.info("Seeds already visited. Forcing seeds into queue...")
        for url in seed_urls:
            queue.put_nowait(url)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    file_lock = asyncio.Lock()
    stats = {'saved': saved_count, 'max_pages': max_pages}

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }

    # CRITICAL FIX: ssl=False prevents immediate crashes on self-signed university certs
    connector = aiohttp.TCPConnector(ssl=False, limit=MAX_CONCURRENT_REQUESTS)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as outfile:
        async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
            workers = [
                asyncio.create_task(crawl_worker(queue, session, semaphore, visited, file_lock, outfile, stats))
                for _ in range(MAX_CONCURRENT_REQUESTS)
            ]
            queue_task = asyncio.create_task(queue.join())
            
            while stats['saved'] < max_pages:
                if queue_task.done() and queue.empty():
                    logging.info("Queue empty. Exhausted all discoverable links.")
                    break
                await asyncio.sleep(2)
                
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

    logging.info(f"Done! Reached {stats['saved']} pages.")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(max_pages=20000))
