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
from requests.utils import parse_http_date # For parsing Last-Modified header date
from typing import Dict, Any, Tuple, Optional

# ---------- Configuration ----------
RAW_DIR = pathlib.Path('data/raw_pages')
RAW_DIR.mkdir(parents=True, exist_ok=True)

RAW_DOCS_DIR = pathlib.Path('data/raw_docs') # Placeholder for future PDF downloads
RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)

SITEMAP_URL = "https://www.gemeentewestland.nl/sitemap.xml"
BASE_URL = "https://www.gemeentewestland.nl/"

DOCUMENT_LINKS_FILE = RAW_DIR / 'document_links.jsonl'
FAILED_URLS_FILE = RAW_DIR / 'failed_urls.txt'
LAST_CRAWL_META_FILE = RAW_DIR / 'last_crawl_meta.json' # NEW: File to store metadata about last crawl

logging.basicConfig(
    filename=RAW_DIR / 'crawl_activity.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger().addHandler(logging.StreamHandler())

DOCUMENT_EXTENSIONS = ('.pdf', '.docx', '.xlsx', '.doc', '.ppt', '.pptx', '.odt', '.ods', '.odp')

# New: Category Mapping based on URL segments
URL_CATEGORY_MAP = {
    "burgerzaken": "Burgerzaken",
    "paspoort-id-kaart-en-rijbewijs": "Paspoorten & Rijbewijzen",
    "bouwen-en-wonen": "Bouwen & Wonen",
    "vergunningen": "Vergunningen",
    "afval": "Afval & Milieu",
    "bedrijven": "Bedrijven",
    "werken-bij": "Werken bij Gemeente",
    "gemeente-info": "Gemeentelijke Informatie",
    "nieuws": "Nieuws",
    "contact": "Contact",
    "organisatie-en-bestuur": "Organisatie & Bestuur",
    "beleid-en-regelgeving": "Beleid & Regelgeving",
    # Add more specific mappings based on your analysis of the website's URL structure
}

# ---------- Helper Functions ----------

def get_category_from_url(url: str) -> str:
    """Infers category based on URL segments."""
    path_segments = [s for s in url.split("://", 1)[-1].split("/") if s] # Clean empty segments
    for segment in path_segments:
        if segment.lower() in URL_CATEGORY_MAP:
            return URL_CATEGORY_MAP[segment.lower()]
    # Fallback to a broader category if no specific match
    if "gemeente" in url.lower() or "over-ons" in url.lower():
        return "Gemeentelijke Informatie"
    return "Algemeen" # Default category if no specific mapping applies

def get_urls_from_sitemap(sitemap_url: str) -> list[tuple[str, str | None]]:
    """
    Fetches URLs and their lastmod dates from a sitemap.
    Handles sitemap indexes recursively.
    Returns a list of (url, lastmod_date_str) tuples.
    """
    urls_with_lastmod = []
    try:
        response = requests.get(sitemap_url, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        # Check for sitemap index (sitemap of sitemaps)
        ns = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        if "sitemapindex" in root.tag:
            for sitemap_elem in root.findall("sitemap:sitemap", ns):
                loc_elem = sitemap_elem.find("sitemap:loc", ns)
                if loc_elem is not None and loc_elem.text:
                    logging.info(f"Found sitemap index, crawling sub-sitemap: {loc_elem.text}")
                    urls_with_lastmod.extend(get_urls_from_sitemap(loc_elem.text)) # Recursively call
        elif "urlset" in root.tag:
            for url_elem in root.findall("sitemap:url", ns):
                loc_elem = url_elem.find("sitemap:loc", ns)
                lastmod_elem = url_elem.find("sitemap:lastmod", ns)
                if loc_elem is not None and loc_elem.text:
                    lastmod_str = lastmod_elem.text if lastmod_elem is not None else None
                    urls_with_lastmod.append((loc_elem.text, lastmod_str))
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching sitemap {sitemap_url}: {e}")
    except ET.ParseError as e:
        logging.error(f"Error parsing sitemap XML {sitemap_url}: {e}")
    return urls_with_lastmod

def save_document_link(doc_url: str, referring_url: str) -> None:
    """Appends identified document links to a JSONL file."""
    record = {
        "document_url": doc_url,
        "referring_url": referring_url,
        "timestamp": datetime.now(timezone.utc).isoformat() # Use UTC for consistency
    }
    with open(DOCUMENT_LINKS_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

def log_failed_url(url: str, reason: str) -> None:
    """Logs URLs that failed to crawl."""
    with open(FAILED_URLS_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{url} | {reason}\n")
    logging.warning(f"Failed to crawl {url}: {reason}")

def get_filename_from_url(url: str) -> str:
    """Generates a consistent filename from a URL."""
    # Simple hash of the URL to ensure unique and valid filenames
    return hashlib.sha256(url.encode('utf-8')).hexdigest() + '.json'

def load_last_crawl_meta() -> Dict[str, Dict[str, Any]]:
    """Loads metadata from previous crawl for incremental checks."""
    if LAST_CRAWL_META_FILE.exists():
        try:
            with open(LAST_CRAWL_META_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding {LAST_CRAWL_META_FILE}: {e}. Starting fresh.")
            return {}
    return {}

def save_last_crawl_meta(meta_data: Dict[str, Dict[str, Any]]) -> None:
    """Saves current crawl metadata."""
    with open(LAST_CRAWL_META_FILE, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=2)

# ---------- Main Crawling Logic ----------

def main():
    logging.info("Starting crawling process...")

    # Load existing metadata from the last crawl
    # This dictionary will map URL to its stored last_modified_header and file_exists status
    last_crawl_meta = load_last_crawl_meta()
    
    # This will store meta for the *current* crawl to be saved at the end
    current_crawl_meta = {}

    sitemap_urls_with_lastmod = get_urls_from_sitemap(SITEMAP_URL)
    logging.info(f"Found {len(sitemap_urls_with_lastmod)} URLs in sitemap(s).")

    # Use a set to keep track of URLs we've processed in this run
    # This helps identify deleted pages later (not implemented in this script but crucial for sync)
    processed_urls_in_this_run = set()

    for url, sitemap_lastmod_str in tqdm(sitemap_urls_with_lastmod, desc="Crawling URLs"):
        if not url.startswith(BASE_URL): # Ensure we only crawl our domain
            continue

        file_name = get_filename_from_url(url)
        path = RAW_DIR / file_name

        processed_urls_in_this_run.add(url)

        # Check if the page exists and if it needs to be re-downloaded
        needs_download = True
        stored_meta = last_crawl_meta.get(url)

        if stored_meta and path.exists():
            stored_last_modified_header = stored_meta.get("last_modified_header")

            try:
                # Use HEAD request to get current Last-Modified header efficiently
                head_response = requests.head(url, timeout=5)
                head_response.raise_for_status()
                current_last_modified_header = head_response.headers.get('Last-Modified')

                # If sitemap lastmod is available and newer, prioritize it
                if sitemap_lastmod_str:
                    sitemap_dt = parse_http_date(sitemap_lastmod_str)
                    if sitemap_dt and stored_last_modified_header:
                        stored_dt = parse_http_date(stored_last_modified_header)
                        if sitemap_dt <= stored_dt: # Sitemap is not newer than stored
                            needs_download = False
                    elif sitemap_dt: # Sitemap has lastmod but no stored header
                        # Assume if sitemap has a date, we might need to check content.
                        # For simplicity here, we proceed to GET if no stored header.
                        pass
                
                # If sitemap not definitive, rely on HTTP Last-Modified header
                if needs_download and current_last_modified_header and stored_last_modified_header:
                    if parse_http_date(current_last_modified_header) <= parse_http_date(stored_last_modified_header):
                        needs_download = False
            except requests.exceptions.RequestException as e:
                logging.warning(f"HEAD request failed for {url}: {e}. Will attempt full GET.")
                # If HEAD fails, assume it needs a full GET, or mark as failed if GET also fails.
                needs_download = True
            except Exception as e:
                logging.warning(f"Error checking Last-Modified for {url}: {e}. Proceeding with GET.")
                needs_download = True

        if not needs_download:
            logging.info(f"Skipping (not modified): {url}")
            # Update current_crawl_meta with old metadata
            if stored_meta:
                current_crawl_meta[url] = stored_meta
            else: # Should not happen if needs_download is False based on stored_meta
                current_crawl_meta[url] = {"last_modified_header": current_last_modified_header, "category": get_category_from_url(url)}
            continue

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            html_content = response.text
            
            # Extract Last-Modified header from the actual GET response
            actual_last_modified_header = response.headers.get('Last-Modified')
            inferred_category = get_category_from_url(url)

            # Save HTML content and metadata
            record = {
                "url": url,
                "html": html_content,
                "timestamp": datetime.now(timezone.utc).isoformat(), # Use UTC
                "last_modified_header": actual_last_modified_header, # Store the actual header
                "category": inferred_category # Store inferred category
            }
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            logging.info(f"Successfully crawled and saved: {url}")
            tqdm.write(f"✅ Crawled: {url} (Category: {inferred_category}, Last-Modified: {actual_last_modified_header})")

            # Update current_crawl_meta
            current_crawl_meta[url] = {
                "last_modified_header": actual_last_modified_header,
                "category": inferred_category
            }

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

        except requests.exceptions.RequestException as e:
            log_failed_url(url, f"Network/HTTP error: {e}")
        except Exception as e:
            log_failed_url(url, f"Error processing: {e}")

    # Save the metadata from this crawl run for the next time
    save_last_crawl_meta(current_crawl_meta)
    
    logging.info("Crawling process finished.")

if __name__ == "__main__":
    main()