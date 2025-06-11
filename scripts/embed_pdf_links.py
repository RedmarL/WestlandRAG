import json
import hashlib # NEW: For generating stable IDs
import os
import pathlib # NEW: For path operations
import logging # For better logging

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm # NEW: For progress bar

# ---------- Configuration ----------
# RAW_PAGES_DIR is where document_links.jsonl is located
RAW_PAGES_DIR = pathlib.Path('data/raw_pages')
DOCUMENT_LINKS_FILE = RAW_PAGES_DIR / 'document_links.jsonl'

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

COLLECTION_NAME_PDF_LINKS = "westland-pdf-links" # New collection for PDF URLs
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" # Match your main embedding model if possible

# Configure logging
logging.basicConfig(
    filename=RAW_PAGES_DIR / 'embed_pdf_links_activity.log', # Log to the raw_pages directory
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger().addHandler(logging.StreamHandler()) # Also print to console

# ---------- Main Embedding Logic ----------
def main():
    logging.info("Starting embedding of PDF links into Qdrant.")

    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logging.info(f"Qdrant client and embedding model '{EMBEDDING_MODEL_NAME}' initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize Qdrant client or SentenceTransformer model: {e}", exc_info=True)
        return

    unique_doc_urls = set()
    if DOCUMENT_LINKS_FILE.exists():
        logging.info(f"Reading document links from {DOCUMENT_LINKS_FILE}...")
        with open(DOCUMENT_LINKS_FILE, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line)
                    doc_url = record.get("document_url")
                    if doc_url:
                        unique_doc_urls.add(doc_url)
                    else:
                        logging.warning(f"Line {line_num} in {DOCUMENT_LINKS_FILE} missing 'document_url': {line.strip()}")
                except json.JSONDecodeError as e:
                    logging.error(f"Skipping malformed JSON line {line_num} in {DOCUMENT_LINKS_FILE}: {line.strip()} - Error: {e}")
                except Exception as e:
                    logging.error(f"Unexpected error processing line {line_num} in {DOCUMENT_LINKS_FILE}: {line.strip()} - Error: {e}", exc_info=True)
    else:
        logging.error(f"Document links file not found: {DOCUMENT_LINKS_FILE}. Please run crawl.py first.")
        return

    if not unique_doc_urls:
        logging.info("No PDF links found to embed. Exiting.")
        return

    urls_list = list(unique_doc_urls)
    logging.info(f"Processing {len(urls_list)} unique PDF links for embedding.")

    # You could potentially enrich the text embedded here if you have a way to
    # extract titles or short descriptions for each PDF link (e.g., from the
    # referring HTML page's <a> tag text, or a separate URL summarization step).
    # For now, we embed the URL itself.
    texts_to_embed = urls_list

    vectors = model.encode(texts_to_embed, show_progress_bar=True).tolist()
    logging.info(f"Generated {len(vectors)} embeddings.")

    points = []
    for i, url in enumerate(urls_list):
        # Generate a stable ID using a hash of the URL
        point_id = hashlib.md5(url.encode('utf-8')).hexdigest()

        # You can add more metadata here if you can reliably extract it
        # (e.g., from referring HTML page or by performing a HEAD request to the PDF)
        payload = {
            "url": url,
            "document_type": "pdf_link", # Differentiate from HTML chunks
            "source_url": url # This link IS the source
            # Add 'title' or 'description' if you obtain it
        }
        points.append(
            models.PointStruct(
                id=point_id, # Use a stable hash as ID
                vector=vectors[i],
                payload=payload
            )
        )

    # Recreate collection - use this for development/initial setup.
    # In production, consider `client.get_collection` and `client.upsert`
    # without `recreate_collection` to avoid data loss on updates.
    try:
        client.recreate_collection(
            collection_name=COLLECTION_NAME_PDF_LINKS,
            vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE)
        )
        logging.info(f"Collection '{COLLECTION_NAME_PDF_LINKS}' recreated.")
    except Exception as e:
        logging.error(f"Failed to recreate Qdrant collection '{COLLECTION_NAME_PDF_LINKS}': {e}", exc_info=True)
        return

    # Upsert points
    try:
        client.upsert(collection_name=COLLECTION_NAME_PDF_LINKS, points=points, wait=True)
        logging.info(f"Successfully embedded and upserted {len(points)} PDF links into '{COLLECTION_NAME_PDF_LINKS}'.")
    except Exception as e:
        logging.error(f"Error during Qdrant upsert for collection '{COLLECTION_NAME_PDF_LINKS}': {e}", exc_info=True)

if __name__ == "__main__":
    main()