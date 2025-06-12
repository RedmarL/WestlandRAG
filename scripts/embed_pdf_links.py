import json
import hashlib
import os
import pathlib
import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models
# from sentence_transformers import SentenceTransformer # REMOVE THIS LINE
from openai import OpenAI # NEW: Import OpenAI client
from dotenv import load_dotenv # NEW: Import dotenv
from tqdm import tqdm

# ---------- Configuration ----------
RAW_PAGES_DIR = pathlib.Path('data/raw_pages')
DOCUMENT_LINKS_FILE = RAW_PAGES_DIR / 'document_links.jsonl'

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

COLLECTION_NAME_PDF_LINKS = "westland-pdf-links"

# NEW: Use OpenAI's embedding model and its dimensions
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Ensure API key is loaded
EMBEDDING_MODEL_NAME = "text-embedding-3-large"
EMBED_DIM = 3072 # Set the dimension explicitly for text-embedding-3-large

# Configure logging
logging.basicConfig(
    filename=RAW_PAGES_DIR / 'embed_pdf_links_activity.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger().addHandler(logging.StreamHandler())

# ---------- Main Embedding Logic ----------
def main():
    logging.info("Starting embedding of PDF links into Qdrant.")

    # NEW: Load environment variables
    load_dotenv()
    global OPENAI_API_KEY # Use global to ensure it's accessible
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        logging.critical("OPENAI_API_KEY environment variable not set. Please set it before running the script.")
        return

    try:
        # NEW: Initialize OpenAI client
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        logging.info("Qdrant client and OpenAI client initialized.")

        pdf_links = []
        if DOCUMENT_LINKS_FILE.exists():
            with open(DOCUMENT_LINKS_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        pdf_links.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        logging.warning(f"Skipping malformed line in {DOCUMENT_LINKS_FILE}: {line.strip()} - {e}")
            logging.info(f"Loaded {len(pdf_links)} PDF links from {DOCUMENT_LINKS_FILE}.")
        else:
            logging.critical(f"Document links file not found at {DOCUMENT_LINKS_FILE}. Please run crawl.py first.")
            return

        if not pdf_links:
            logging.info("No PDF links to embed. Exiting.")
            return

        # Prepare descriptions for embedding
        pdf_texts = [link.get('description', link['url']) for link in pdf_links] # Use description if available, else URL

        logging.info(f"Generating embeddings for {len(pdf_texts)} PDF links with {EMBEDDING_MODEL_NAME}...")
        
        # NEW: Generate embeddings using OpenAI API
        # The OpenAI API can handle lists of texts for batch embedding
        embeddings_response = openai_client.embeddings.create(
            input=pdf_texts,
            model=EMBEDDING_MODEL_NAME,
            dimensions=EMBED_DIM # Specify the exact dimension
        )
        vectors = [data.embedding for data in embeddings_response.data]
        logging.info("Embeddings generated.")

        points = []
        for i, link_data in enumerate(tqdm(pdf_links, desc="Preparing points for Qdrant")):
            url = link_data['url']
            source_url = link_data.get('source_url', 'unknown') # URL of the page where the PDF link was found
            point_id = int(hashlib.sha256(url.encode('utf-8')).hexdigest(), 16) % (10**10) # Stable ID

            payload = {
                "document_type": "pdf_link",
                "source_url": url, # This link IS the source
                "found_on_page": source_url, # The page it was found on
                "description": link_data.get('description', '') # The description generated (if any)
            }
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vectors[i],
                    payload=payload
                )
            )

        try:
            # Recreate collection with the new dimension
            client.recreate_collection(
                collection_name=COLLECTION_NAME_PDF_LINKS,
                vectors_config=models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE)
            )
            logging.info(f"Collection '{COLLECTION_NAME_PDF_LINKS}' recreated with {EMBED_DIM} dimensions.")
        except Exception as e:
            logging.error(f"Failed to recreate Qdrant collection '{COLLECTION_NAME_PDF_LINKS}': {e}", exc_info=True)
            return

        try:
            client.upsert(collection_name=COLLECTION_NAME_PDF_LINKS, points=points, wait=True)
            logging.info(f"Successfully embedded and upserted {len(points)} PDF links into '{COLLECTION_NAME_PDF_LINKS}'.")
        except Exception as e:
            logging.error(f"Failed to upsert points into '{COLLECTION_NAME_PDF_LINKS}': {e}", exc_info=True)

    except Exception as e:
        logging.critical(f"An unexpected error occurred during PDF link embedding: {e}", exc_info=True)

if __name__ == "__main__":
    main()