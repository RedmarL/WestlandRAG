import requests
import pathlib
import time
import json
import xml.etree.ElementTree as ET
import hashlib
import logging
from datetime import datetime
from tqdm import tqdm
from bs4 import BeautifulSoup

# ---------- Configuration ----------
RAW_DIR = pathlib.Path('data/raw_pages')
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Files to log found document links and failed URLs
DOCUMENT_LINKS_FILE = RAW_DIR / 'document_links.jsonl'
FAILED_URLS_FILE = RAW_DIR / 'failed_urls.txt'

SITEMAP_URL = "https://www.gemeentewestland.nl/sitemap.xml"
BASE_URL = "https://www.gemeentewestland.nl/" # Ensure this matches for slug generation

# Configure logging
logging.basicConfig(
    filename=RAW_DIR / 'crawl_activity.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger().addHandler(logging.StreamHandler()) # Also print to console

# Types of documents we'll identify links for
DOCUMENT_EXTENSIONS = ('.pdf', '.docx', '.xlsx', '.doc', '.ppt', '.pptx', '.odt', '.ods', '.odp')

# ---------- Helper Functions ----------

def get_slug_from_url(url: str) -> str:
    """Generates a consistent filename slug from a URL."""
    # Remove protocol (http/https) and www.
    clean_url = url.replace('https://', '').replace('http://', '').replace('www.', '')
    # Replace slashes with underscores, remove trailing slash
    slug = clean_url.replace('/', '_').rstrip('_')
    # Remove query parameters and anchors
    slug = slug.split('?')[0].split('#')[0]
    # Shorten if too long (optional, but good for file systems)
    if len(slug) > 200:
        hash_suffix = hashlib.md5(url.encode('utf-8')).hexdigest()[:8]
        slug = slug[:190] + '_' + hash_suffix
    return slug

def log_failed_url(url: str, reason: str) -> None:
    """Logs URLs that failed to crawl."""
    with open(FAILED_URLS_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{url} | {reason}\n")
    logging.warning(f"Failed to crawl {url}: {reason}")

def save_document_link(doc_url: str, referring_url: str) -> None:
    """Saves identified document links to a JSONL file."""
    record = {
        "document_url": doc_url,
        "referring_url": referring_url,
        "timestamp": datetime.now().isoformat()
    }
    with open(DOCUMENT_LINKS_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
    logging.info(f"Identified document link: {doc_url} (from {referring_url})")

def get_urls_from_sitemap(sitemap_url: str) -> list[str]:
    """Fetches URLs from a sitemap.xml."""
    urls = []
    try:
        response = requests.get(sitemap_url, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        for url_element in root.findall('{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
            loc = url_element.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
            if loc is not None and loc.text and loc.text.startswith(BASE_URL):
                urls.append(loc.text)
        logging.info(f"Found {len(urls)} URLs in sitemap: {sitemap_url}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching sitemap {sitemap_url}: {e}")
    except ET.ParseError as e:
        logging.error(f"Error parsing sitemap XML {sitemap_url}: {e}")
    return urls

# ---------- Main Crawler Logic ----------
def main():
    logging.info("Starting web crawling process.")

    all_urls = get_urls_from_sitemap(SITEMAP_URL)
    if not all_urls:
        logging.error("No URLs found in sitemap. Exiting.")
        return

    # Check for existing crawled files to avoid re-crawling
    crawled_files = {path.stem for path in RAW_DIR.glob('*.json')}
    logging.info(f"Found {len(crawled_files)} previously crawled HTML files.")

    # Note: DOCUMENT_LINKS_FILE is always appended to, so no need to clear it here.
    # If you want to overwrite it each run, uncomment the next lines:
    # if DOCUMENT_LINKS_FILE.exists():
    #     DOCUMENT_LINKS_FILE.unlink() # Removes the file

    for url in tqdm(all_urls, desc="Crawling URLs"):
        slug = get_slug_from_url(url)
        path = RAW_DIR / f'{slug}.json'

        if path.stem in crawled_files:
            logging.info(f"Skipping (already exists): {url}")
            continue

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            html_content = response.text

            # Save HTML content
            record = {
                "url": url,
                "html": html_content,
                "timestamp": datetime.now().isoformat()
            }
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            logging.info(f"Successfully crawled and saved: {url}")

            # Parse HTML to find links to other documents (PDFs, DOCX, etc.)
            soup = BeautifulSoup(html_content, 'html.parser')
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                # Resolve relative URLs
                if not href.startswith('http'):
                    href = requests.compat.urljoin(url, href)

                # Check if it's a document link and within the same domain (or specific allowed domains)
                if any(href.lower().endswith(ext) for ext in DOCUMENT_EXTENSIONS) and href.startswith(BASE_URL):
                    save_document_link(href, url)

            time.sleep(0.5)  # Be polite to the server

        except requests.exceptions.RequestException as e:
            log_failed_url(url, f"Network/HTTP error: {e}")
        except Exception as e: # Catch any other unexpected errors during processing
            log_failed_url(url, f"Unexpected error: {e}")

    logging.info("Web crawling process finished.")

if __name__ == "__main__":
    main()