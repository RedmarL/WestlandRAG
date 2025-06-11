import pathlib
import json
import time
import requests
from typing import List, Dict, Union, Any
from tqdm import tqdm
from transformers import AutoTokenizer
from bs4 import BeautifulSoup, Tag
import uuid # For generating unique IDs
from datetime import datetime # For timestamps
import logging # For better logging

# ---------- Configuration ----------
# Use RAW_PAGES_DIR for consistency, as per crawl.py
RAW_PAGES_DIR = pathlib.Path("data/raw_pages")
CHUNK_DIR = pathlib.Path("data/chunks")
CHUNK_DIR.mkdir(exist_ok=True)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

MAX_TOKENS = 512
OVERLAP = 100

# Configure logging
logging.basicConfig(
    filename=CHUNK_DIR / 'chunking_activity.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger().addHandler(logging.StreamHandler()) # Also print to console

# ---------- Helper Functions ----------

def extract_content_from_html(html_content: str) -> Dict[str, Any]:
    """
    Parses HTML to extract structured content blocks (headings, paragraphs, lists, tables)
    along with the main document title.
    Focuses on main content elements and tries to filter out common boilerplate.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Get document title from <title> tag
    document_title = soup.title.string if soup.title and soup.title.string else "Onbekend document"

    content_blocks = []

    # Attempt to find the main content area. This is highly dependent on website structure.
    # Common patterns: <main> tag, div with specific ID/class (e.g., 'main-content', 'article-body')
    # For a general approach, we'll iterate common tags, but be aware of noise.
    main_content_div = soup.find('main') or soup.find('div', class_='main-content') or soup.find('article')

    # If a specific main content area is found, limit search to that area
    if main_content_div:
        search_scope = main_content_div
    else:
        search_scope = soup # Fallback to entire soup if main content area not found

    # Iterate through common content-bearing tags
    for tag in search_scope.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'table', 'div']):
        # Basic filtering to exclude common non-content elements
        # This list might need to be expanded based on specific website's HTML structure
        if tag.name == 'div' and (
            'navbar' in tag.get('class', []) or 'header' in tag.get('id', '') or
            'footer' in tag.get('id', '') or 'sidebar' in tag.get('class', []) or
            'navigation' in tag.get('class', []) or 'cookie-notice' in tag.get('class', [])
        ):
            continue

        if tag.name == 'table':
            # For tables, convert to a simplified text representation
            table_text = []
            for row in tag.find_all('tr'):
                row_data = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
                if row_data:
                    # Filter out empty rows or rows with only boilerplate (e.g., hidden formatting)
                    cleaned_row_data = [d for d in row_data if d.strip()]
                    if cleaned_row_data:
                        table_text.append("|".join(cleaned_row_data))
            if table_text:
                content_blocks.append({"type": "table", "text": "\n".join(table_text)})
        else:
            text = tag.get_text(separator=' ', strip=True)
            if text:
                content_blocks.append({"type": tag.name, "text": text})

    return {"document_title": document_title, "content_blocks": content_blocks}


def split_content_blocks_by_tokens(
    document_data: Dict[str, Any], max_tokens: int, overlap: int, doc_metadata: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Splits document content blocks into chunks based on token limits with overlap.
    Includes document-level metadata and a unique ID for each chunk.
    """
    chunks = []
    current_chunk_text = ""
    current_chunk_tokens = 0

    document_title = document_data.get("document_title", "Onbekend document")
    content_blocks = document_data.get("content_blocks", [])

    for block in content_blocks:
        block_text = block["text"]
        block_tokens = len(tokenizer.encode(block_text))

        # Check if adding this block exceeds max_tokens
        if current_chunk_tokens + block_tokens <= max_tokens:
            current_chunk_text += (" " + block_text if current_chunk_text else block_text)
            current_chunk_tokens += block_tokens
        else:
            # If current chunk is not empty, save it
            if current_chunk_text:
                chunk_id = str(uuid.uuid4()) # Generate UUID for this chunk
                chunk_metadata = {
                    "id": chunk_id, # Use UUID as chunk ID
                    "text": current_chunk_text,
                    "document_title": document_title,
                    **doc_metadata, # Include all document-level metadata
                    "chunk_idx": len(chunks) # Index of this chunk within its document
                }
                chunks.append(chunk_metadata)

            # Start a new chunk, potentially with overlap
            if overlap > 0 and current_chunk_text:
                # Ensure we don't try to overlap more tokens than available in current_chunk_text
                encoded_current_chunk = tokenizer.encode(current_chunk_text)
                actual_overlap_tokens = min(overlap, len(encoded_current_chunk))
                overlap_text = tokenizer.decode(encoded_current_chunk[-actual_overlap_tokens:])
                
                current_chunk_text = overlap_text + " " + block_text
                current_chunk_tokens = len(tokenizer.encode(current_chunk_text))
            else:
                current_chunk_text = block_text
                current_chunk_tokens = block_tokens

    # Add the last chunk if it's not empty
    if current_chunk_text:
        chunk_id = str(uuid.uuid4()) # Generate UUID for the last chunk too
        chunk_metadata = {
            "id": chunk_id, # Use UUID as chunk ID
            "text": current_chunk_text,
            "document_title": document_title,
            **doc_metadata,
            "chunk_idx": len(chunks)
        }
        chunks.append(chunk_metadata)

    return chunks

# ---------- Main Processing Loop ----------
def main():
    logging.info("Starting HTML chunking process.")

    html_files = list(RAW_PAGES_DIR.glob('*.json'))
    if not html_files:
        logging.error(f"No HTML files found in {RAW_PAGES_DIR}. Please run crawl.py first.")
        return

    logging.info(f"Processing {len(html_files)} HTML files for chunking...")

    for file_path in tqdm(html_files, desc="Chunking HTML files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            source_url = data.get("url")
            html_content = data.get("html")
            original_file_name = file_path.name # Use the slug filename here
            last_modified_date = datetime.now().isoformat() # This timestamp is when the HTML was crawled

            if not source_url or not html_content:
                logging.warning(f"Skipped {file_path.name}: Missing URL or HTML content.")
                continue

            # Default metadata for HTML pages.
            # department: Could be inferred from URL path segments or page content if logic is added.
            doc_metadata = {
                "source": source_url, # Renamed source_url to source for consistency with Qdrant payload
                "document_type": "html_page",
                "department": "Algemeen", # Placeholder, improve if possible
                "last_modified_date": last_modified_date,
                "original_file_name": original_file_name
            }

            parsed_data = extract_content_from_html(html_content)
            content_blocks = parsed_data["content_blocks"]
            document_title = parsed_data["document_title"]

            if not content_blocks:
                logging.info(f"Skipped (no significant content blocks extracted): {file_path.name}")
                continue

            chunks = split_content_blocks_by_tokens(
                {"document_title": document_title, "content_blocks": content_blocks},
                MAX_TOKENS,
                OVERLAP,
                doc_metadata
            )

            # Write out each chunk as a separate JSON file
            for chunk in chunks: # Iterate directly over chunks, no need for idx here for filename
                # Use the UUID from the chunk's payload as part of the filename for uniqueness and traceability
                out_path = CHUNK_DIR / f"{file_path.stem}_{chunk['id']}.json"
                json.dump(chunk, out_path.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
            logging.info(f"Processed {file_path.name} → {len(chunks)} chunks.")

        except json.JSONDecodeError:
            logging.error(f"Failed to parse JSON from {file_path.name}")
        except Exception as e:
            logging.error(f"Error processing {file_path.name}: {e}", exc_info=True) # exc_info to log traceback

    logging.info("HTML chunking process complete.")

if __name__ == "__main__":
    main()