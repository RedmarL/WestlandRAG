import requests
import pathlib
import time
import json
import xml.etree.ElementTree as ET
import hashlib
import logging
from datetime import datetime, timezone
from tqdm import tqdm
from bs4 import BeautifulSoup
# CORRECTED IMPORT: Use email.utils for parsing HTTP dates if needed
# from requests.utils import parse_http_date # This caused the ImportError
from email.utils import parsedate_to_datetime # Use this for parsing HTTP dates
from typing import Dict, Any, Tuple, Optional, List

# ---------- Configuration ----------
RAW_DIR = pathlib.Path('data/raw_pages')
RAW_DIR.mkdir(parents=True, exist_ok=True)

RAW_DOCS_DIR = pathlib.Path('data/raw_docs') # Placeholder for future PDF downloads
RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)

SITEMAP_URL = "https://www.gemeentewestland.nl/sitemap.xml"
BASE_URL = "https://www.gemeentewestland.nl/"

DOCUMENT_LINKS_FILE = RAW_DIR / 'document_links.jsonl'
FAILED_URLS_FILE = RAW_DIR / 'failed_urls.txt'
LAST_CRAWL_META_FILE = RAW_DIR / 'last_crawl_meta.json' # File to store metadata about last crawl

logging.basicConfig(
    filename=RAW_DIR / 'crawl_activity.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger().addHandler(logging.StreamHandler()) # Also print to console

DOCUMENT_EXTENSIONS = ('.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx') # Common document extensions

# NEW: Define the URL to user-friendly category mapping
# This map should be populated based on unique top-level URL segments found after a full crawl.
# For example, if you find '/onderwijs/', add 'onderwijs': 'Onderwijs'.
# The values are the human-readable category names.
URL_CATEGORY_MAP: Dict[str, str] = {
    "aanvragen-en-regelen": "Aanvragen & Regelen",
    "burgerzaken": "Burgerzaken",
    "paspoort-id-kaart-en-rijbewijs": "Paspoorten & Rijbewijzen", # This specific segment might not be a top-level category
    "bouwen-en-wonen": "Bouwen & Wonen",
    "vergunningen": "Vergunningen",
    "afval": "Afval & Milieu", # This would map if 'afval' was a top-level segment
    "bedrijven": "Bedrijven & Ondernemen",
    "werken-bij": "Werken bij Gemeente",
    "gemeente-info": "Gemeentelijke Informatie",
    "nieuws": "Nieuws & Publicaties",
    "contact": "Contact",
    "organisatie-en-bestuur": "Organisatie & Bestuur",
    "beleid-en-regelgeving": "Beleid & Regelgeving",
    "oekraine": "Oekraïne", # Added based on your sitemap example
    "cultuur-recreatie": "Cultuur & Recreatie", # Example of another common category
    "zorg-en-welzijn": "Zorg & Welzijn", # Example
    "overig": "Overig", # Explicitly map the default category
    "pagina-niet-gevonden": "Systeempagina", # Handle known system/error pages
    # IMPORTANT: You will need to expand this map after identifying all unique
    # first-level URL segments from a comprehensive crawl of the website.
    # Any segment not in this map will use its raw name as the category.
}


# ---------- helpers ----------

def get_urls_from_sitemap(sitemap_url: str) -> List[str]:
    """Fetches and parses URLs from a sitemap."""
    urls = []
    try:
        response = requests.get(sitemap_url, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        for url_element in root.findall('{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
            loc_element = url_element.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
            if loc_element is not None and loc_element.text:
                urls.append(loc_element.text)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching sitemap {sitemap_url}: {e}")
    except ET.ParseError as e:
        logging.error(f"Error parsing sitemap XML from {sitemap_url}: {e}")
    return urls

def load_last_crawl_meta() -> Dict[str, str]:
    """Loads metadata from the last crawl."""
    if LAST_CRAWL_META_FILE.exists():
        try:
            with open(LAST_CRAWL_META_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logging.warning(f"Could not decode {LAST_CRAWL_META_FILE}, starting fresh: {e}")
            return {}
    return {}

def save_last_crawl_meta(meta: Dict[str, str]):
    """Saves metadata for the current crawl."""
    with open(LAST_CRAWL_META_FILE, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

def log_failed_url(url: str, error: str):
    """Logs URLs that failed to crawl."""
    with open(FAILED_URLS_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{url} - {error}\n")
    logging.warning(f"Failed to crawl {url}: {error}")

def save_document_link(url: str, source_url: str):
    """Saves links to external documents (e.g., PDFs)."""
    with open(DOCUMENT_LINKS_FILE, 'a', encoding='utf-8') as f:
        json.dump({"url": url, "source_url": source_url, "timestamp": datetime.now(timezone.utc).isoformat()}, f, ensure_ascii=False)
        f.write('\n')

def get_hash_for_content(content: str) -> str:
    """Generates a SHA256 hash for content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

# MODIFIED: infer_category_from_url now uses URL_CATEGORY_MAP
def infer_category_from_url(url: str) -> str:
    """
    Infers a category from the URL path and maps it to a user-friendly name
    using the predefined URL_CATEGORY_MAP.
    """
    path_parts = url.replace(BASE_URL, '').strip('/').split('/')
    
    # Get the raw top-level segment
    raw_segment = ""
    if path_parts and path_parts[0]:
        raw_segment = path_parts[0]
    
    # Look up in the URL_CATEGORY_MAP.
    # If a mapping exists, return the mapped value.
    # Otherwise, return the raw segment itself.
    # Handle the case where raw_segment is empty (e.g., for the base URL)
    if raw_segment:
        return URL_CATEGORY_MAP.get(raw_segment, raw_segment)
    
    # Default category for the base URL or if no segment is found
    # This also uses the map, so "overig" will be "Overig" if mapped.
    return URL_CATEGORY_MAP.get("overig", "Overig")


# ---------- Main Crawling Logic ----------

def crawl_website():
    logging.info("Starting website crawl...")

    # Load metadata from previous crawl to determine if content has changed
    last_crawl_meta = load_last_crawl_meta()
    current_crawl_meta = {} # To store metadata for the current crawl

    # Get all URLs from the sitemap
    urls_to_crawl = get_urls_from_sitemap(SITEMAP_URL)
    logging.info(f"Found {len(urls_to_crawl)} URLs in the sitemap.")

    if not urls_to_crawl:
        logging.info("No URLs to process (sitemap was empty).")
        return

    # Clear previous failed URLs and document links at the start of a new full crawl
    # NOTE: If you want to append, remove these two lines. For a fresh crawl, keep them.
    if FAILED_URLS_FILE.exists():
        FAILED_URLS_FILE.unlink()
    if DOCUMENT_LINKS_FILE.exists():
        DOCUMENT_LINKS_FILE.unlink()

    with tqdm(total=len(urls_to_crawl), desc="Crawling URLs") as pbar:
        for url in urls_to_crawl:
            # We pass the last_modified_info for the specific URL to crawl_url
            last_modified_info_for_url = last_crawl_meta.get(url, {})
            crawled_content_hash = crawl_url(url, last_modified_info_for_url)

            if crawled_content_hash:
                # Content was newly crawled or changed. Store its new hash.
                # The category inferred here will be the mapped, user-friendly one.
                inferred_category = infer_category_from_url(url)
                
                # Get the actual last-modified header received, or use current time if not present
                # This ensures the metadata accurately reflects the last successful modification time.
                actual_last_modified_header = last_modified_info_for_url.get("last_modified_header", datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT"))

                current_crawl_meta[url] = {
                    "content_hash": crawled_content_hash,
                    "category": inferred_category, # This now contains the mapped category
                    "last_modified_header": actual_last_modified_header # Store the header for next time
                }
            # If crawl_url returns None, it means either an error or content hasn't changed.
            # In that case, we want to retain the old metadata for that URL if it existed.
            elif url in last_crawl_meta:
                current_crawl_meta[url] = last_crawl_meta[url]
            # If it was a new URL that failed to crawl, it won't be in last_crawl_meta and won't be added to current_crawl_meta, which is fine.

            pbar.update(1)

    # Save the metadata from this crawl run for the next time
    save_last_crawl_meta(current_crawl_meta)
    logging.info("Website crawl complete. Metadata saved.")


def crawl_url(url: str, last_modified_info: Dict[str, str]) -> Optional[str]:
    """
    Crawls a single URL, saves content if new/modified, and extracts links.
    Returns content hash if crawled, otherwise None.
    """
    headers = {}
    
    # Use If-Modified-Since header for incremental crawling
    prev_last_modified_header = last_modified_info.get("last_modified_header")
    if prev_last_modified_header:
        headers['If-Modified-Since'] = prev_last_modified_header
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Check for 304 Not Modified
        if response.status_code == 304:
            logging.info(f"✅ Not Modified: {url}")
            # If server sent 304, it means the content is the same,
            # and the last_modified_header is still valid.
            # We don't return a new hash, indicating no content update needed.
            return None 

        html_content = response.text
        content_hash = get_hash_for_content(html_content)

        # Additional check: If server didn't send 304, but hash is the same,
        # consider it unchanged. This can happen with proxies or caching.
        prev_content_hash = last_modified_info.get("content_hash")
        if prev_content_hash == content_hash:
            logging.info(f"✅ Content hash unchanged for: {url}")
            # If content hash is the same, we still might want to update the stored
            # last_modified_header if the server returned a new one (e.g., re-validation).
            new_last_modified_header = response.headers.get('Last-Modified')
            if new_last_modified_header and new_last_modified_header != prev_last_modified_header:
                 # Update the stored metadata with the new header for consistency
                 last_modified_info["last_modified_header"] = new_last_modified_header
            return None

        # Content is new or modified, save it
        filename_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()
        output_filepath = RAW_DIR / f"{filename_hash}.json"

        # The category inferred here uses the MODIFIED infer_category_from_url
        inferred_category = infer_category_from_url(url)
        
        # Capture the actual Last-Modified header from the response, or current time as fallback
        actual_last_modified_header = response.headers.get('Last-Modified', datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT"))

        data_to_save = {
            "url": url,
            "content": html_content,
            "crawl_timestamp": datetime.now(timezone.utc).isoformat(),
            "last_modified_header": actual_last_modified_header,
            "category": inferred_category, # This will now be the mapped category
            "content_hash": content_hash # Store hash for future checks
        }

        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)

        logging.info(f"✅ Crawled: {url} (Category: {inferred_category}, Last-Modified: {actual_last_modified_header})")

        # Parse HTML to find links to other documents (PDFs, DOCX, etc.)
        soup = BeautifulSoup(html_content, 'html.parser')
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            # Resolve relative URLs
            if not href.startswith('http'):
                href = requests.compat.urljoin(url, href) # Safely join base URL with relative path

            # Check if it's a document link and within the same domain (or specific allowed domains)
            if any(href.lower().endswith(ext) for ext in DOCUMENT_EXTENSIONS) and href.startswith(BASE_URL):
                save_document_link(href, url)

        time.sleep(0.5)  # Be polite to the server
        return content_hash # Indicate that new content was crawled

    except requests.exceptions.RequestException as e:
        log_failed_url(url, f"Network/HTTP error: {e}")
        return None
    except Exception as e:
        log_failed_url(url, f"Error processing: {e}")
        return None

# ---------- Entry point ----------
if __name__ == "__main__":
    crawl_website()