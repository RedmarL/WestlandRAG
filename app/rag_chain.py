import json
import numpy as np
# from sentence_transformers import SentenceTransformer # Keep for KeyBERT, but not for query embeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
# import ollama # REMOVED: No longer using Ollama
import logging
import os
import pathlib
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Set, Optional, Tuple

from keybert import KeyBERT
from stop_words import get_stop_words
from openai import OpenAI # Already imported, will be used for chat now
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer # Keep for KeyBERT, but not for query embeddings

# -------------------- Configuration --------------------
# Paths
DATA_DIR = pathlib.Path('data')
URL_DESCRIPTIONS_FILE = DATA_DIR / 'url_descriptions.json'
URL_VECTORS_FILE = DATA_DIR / 'url_vectors.json'
KEYWORD_SET_FILE = DATA_DIR / 'keyword_set.json'

# Qdrant
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_HTML = "westland-openai-embedding"
QDRANT_COLLECTION_PDF_LINKS = "westland-pdf-links"

# Models
# Define separate models for query embedding (OpenAI) and keyword extraction (SentenceTransformer)
QUERY_EMBEDDING_MODEL_OPENAI = "text-embedding-3-large"
QUERY_EMBED_DIM = 3072 # As per Qdrant error message
KEYWORD_EMBEDDING_MODEL_ST = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" # For KeyBERT

# NEW: OpenAI Chat Model for LLM response
OPENAI_CHAT_MODEL_NAME = "gpt-4o-mini" # You can change this to "gpt-3.5-turbo" or "gpt-4o"

# Logging setup
LOG_FILE = DATA_DIR / 'rag_chain_activity.log'
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger().addHandler(logging.StreamHandler())

# -------------------- Helper Functions --------------------

def bronvermelding_from_source(source_value: str) -> str:
    """Returns a cleaned source URL for citation."""
    return source_value.strip() if source_value else "https://www.gemeentewestland.nl/"

def extract_text(hit: models.ScoredPoint) -> str:
    """Extracts the 'text' payload from a Qdrant hit."""
    return hit.payload.get('text', '')

def extract_url(hit: models.ScoredPoint) -> str:
    """Extracts the 'source' or 'source_url' payload from a Qdrant hit."""
    return hit.payload.get('source', hit.payload.get('source_url', ''))

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def load_keywords(filepath: pathlib.Path) -> Set[str]:
    """Loads a set of keywords from a JSON file."""
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    logging.warning(f"Keyword file not found at {filepath}. Keyword filtering will be skipped.")
    return set()

def filter_by_keywords(hits: List[models.ScoredPoint], query_keywords: Set[str], min_matches: int = 1) -> List[models.ScoredPoint]:
    """Filters Qdrant hits to only include those that share keywords with the query."""
    if not query_keywords:
        return hits

    filtered_hits = []
    for hit in hits:
        hit_keywords = set(hit.payload.get('keywords', []))
        if len(query_keywords.intersection(hit_keywords)) >= min_matches:
            filtered_hits.append(hit)
    return filtered_hits

# -------------------- Main RAG Logic --------------------

async def main_rag_chain_loop():
    logging.info("Starting local RAG chain execution.")

    # Load environment variables
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        logging.critical("OPENAI_API_KEY environment variable not set. Please set it before running the script.")
        return

    # ======== Load models and data =========
    try:
        # Initialize OpenAI client for query embeddings and chat completions
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize SentenceTransformer for KeyBERT only
        keyword_model_st = SentenceTransformer(KEYWORD_EMBEDDING_MODEL_ST)
        kw_extractor = KeyBERT(model=keyword_model_st)

        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        logging.info("Embedding models and Qdrant client initialized.")

        # Verify Qdrant collections exist
        for collection_name in [QDRANT_COLLECTION_HTML, QDRANT_COLLECTION_PDF_LINKS]:
            if not qdrant_client.collection_exists(collection_name=collection_name):
                logging.error(f"Qdrant collection '{collection_name}' does not exist. Please run relevant embedding scripts.")
                return
        logging.info("Qdrant collections verified.")

        url_descriptions = {}
        if URL_DESCRIPTIONS_FILE.exists():
            with open(URL_DESCRIPTIONS_FILE, encoding='utf-8') as f:
                url_descriptions = json.load(f)
            logging.info(f"Loaded {len(url_descriptions)} URL descriptions.")
        else:
            logging.warning(f"URL descriptions file not found at {URL_DESCRIPTIONS_FILE}.")

        url_vectors = {}
        if URL_VECTORS_FILE.exists():
            with open(URL_VECTORS_FILE, encoding='utf-8') as f:
                url_vectors = json.load(f)
            logging.info(f"Loaded {len(url_vectors)} URL vectors.")
        else:
            logging.warning(f"URL vectors file not found at {URL_VECTORS_FILE}. URL similarity search will be skipped.")
        
        keyword_set = load_keywords(KEYWORD_SET_FILE)
        logging.info(f"Loaded {len(keyword_set)} keywords.")

    except Exception as e:
        logging.critical(f"Critical error during setup: {e}", exc_info=True)
        return

    while True:
        vraag = input("\nStel je vraag (typ 'exit' om te stoppen): ")
        if vraag.lower() == 'exit':
            break

        logging.info(f"Received query: '{vraag}'")

        # 1. Embed the user query using OpenAI's model
        logging.info(f"Generating query embedding with {QUERY_EMBEDDING_MODEL_OPENAI}...")
        response = openai_client.embeddings.create(
            input=vraag,
            model=QUERY_EMBEDDING_MODEL_OPENAI,
            dimensions=QUERY_EMBED_DIM
        )
        query_vector = response.data[0].embedding
        logging.info("Query embedded.")

        # 2. Extract keywords from the query using KeyBERT
        query_keywords_list = kw_extractor.extract_keywords(
            vraag,
            keyphrase_ngram_range=(1, 2),
            stop_words=list(get_stop_words('dutch')),
            top_n=5
        )
        query_keywords = {kw[0] for kw in query_keywords_list}
        logging.info(f"Extracted query keywords: {query_keywords}")

        # 3. Retrieve relevant HTML chunks
        logging.info(f"Searching Qdrant collection '{QDRANT_COLLECTION_HTML}' for HTML chunks...")
        html_hits = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_HTML,
            query_vector=query_vector,
            limit=5,
            with_payload=True,
            with_vectors=False
        )
        logging.info(f"Found {len(html_hits)} raw HTML chunks.")

        # Filter HTML chunks by keywords
        filtered_html_hits = filter_by_keywords(html_hits, query_keywords)
        logging.info(f"Found {len(filtered_html_hits)} HTML chunks after keyword filtering.")

        # 4. Retrieve relevant PDF links
        logging.info(f"Searching Qdrant collection '{QDRANT_COLLECTION_PDF_LINKS}' for PDF links...")
        pdf_link_hits = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_PDF_LINKS,
            query_vector=query_vector, # Now also 3072-dim after re-embedding
            limit=3,
            with_payload=True,
            with_vectors=False
        )
        logging.info(f"Found {len(pdf_link_hits)} PDF links.")

        # Prioritize one PDF link if its URL is highly relevant
        prioritized_pdf_url = None
        if url_vectors and pdf_link_hits:
            logging.info("Attempting to prioritize a PDF link based on URL descriptions...")
            pdf_url_vectors_data = []
            for hit in pdf_link_hits:
                pdf_url = hit.payload.get('source_url')
                if pdf_url and pdf_url in url_vectors:
                    pdf_url_vectors_data.append({
                        "url": pdf_url,
                        "vector": url_vectors[pdf_url],
                        "score": hit.score
                    })

            if pdf_url_vectors_data:
                max_url_similarity = -1
                best_url_for_pdf = None

                for pdf_vec_data in pdf_url_vectors_data:
                    sim = cosine_similarity(np.array(query_vector), np.array(pdf_vec_data["vector"]))
                    if sim > max_url_similarity:
                        max_url_similarity = sim
                        best_url_for_pdf = pdf_vec_data["url"]
                
                if best_url_for_pdf and max_url_similarity > 0.5:
                    prioritized_pdf_url = best_url_for_pdf
                    logging.info(f"Prioritized PDF: {prioritized_pdf_url} (Similarity: {max_url_similarity:.2f})")
                else:
                    logging.info("No PDF link highly prioritized based on URL descriptions.")
            else:
                logging.info("No matching URL vectors found for top PDF links.")


        # 5. Prepare context for LLM
        context_chunks = []
        sources = set()

        # Add filtered HTML chunks to context
        for i, hit in enumerate(filtered_html_hits):
            text_content = extract_text(hit)
            source_url = bronvermelding_from_source(hit.payload.get('source'))
            context_chunks.append(f"Context uit website ({source_url}):\n{text_content}")
            sources.add(source_url)
            logging.debug(f"Added HTML chunk from {source_url} (Score: {hit.score:.2f})")
            
            snippet_text = hit.payload.get('text', '')[:200].replace('\n', ' ')
            print(f"  Snippet: {snippet_text}...")

        # Add information about retrieved PDF links
        pdf_source_info = []
        for hit in pdf_link_hits:
            pdf_url = hit.payload.get('source_url')
            pdf_desc = url_descriptions.get(pdf_url, "een relevant document")
            pdf_source_info.append(f"PDF document: {pdf_desc} ({pdf_url})")
            sources.add(pdf_url)
            logging.debug(f"Added PDF link from {pdf_url} (Score: {hit.score:.2f})")

        if pdf_source_info:
            context_chunks.append("\n\nMogelijke relevante PDF-documenten:\n" + "\n".join(pdf_source_info))

        combined_context = "\n\n---\n\n".join(context_chunks)

        # --- NEW DEBUGGING STEP: Print the combined context to the console/log ---
        logging.info("\n--- Combined Context Sent to LLM ---")
        logging.info(combined_context)
        logging.info("--- End Combined Context ---\n")
        # --- END NEW DEBUGGING STEP ---

        if not combined_context:
            antwoord = "Het antwoord op uw vraag is niet terug te vinden in de gevonden informatie."
            logging.info("No relevant context found to generate an answer.")
        else:
            # 6. Generate LLM response using OpenAI
            system_prompt = (
                "Je bent een behulpzame assistent van de Gemeente Westland. "
                "Beantwoord vragen zo goed mogelijk in het Nederlands, in de stijl van een gemeentelijke medewerker. "
                "Antwoord alleen op basis van de verstrekte context. "
                "Als het antwoord niet in de context staat, zeg dan: 'Het antwoord op uw vraag is niet terug te vinden in de gevonden informatie.' "
                "Gebruik altijd 'u' in plaats van 'jij'. "
                "Indien u PDF-documenten vermeldt, verwijs ernaar met hun URL en een korte beschrijving als die beschikbaar is."
            )
            
            user_prompt = (
                f"Context:\n{combined_context}\n\nVraag: {vraag}\n\n"
                "Antwoord:"
            )

            try:
                logging.info(f"Sending prompt to OpenAI chat model '{OPENAI_CHAT_MODEL_NAME}'.")
                response = openai_client.chat.completions.create(
                    model=OPENAI_CHAT_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                antwoord = response.choices[0].message.content.strip()
                logging.info("OpenAI chat response received.")
            except Exception as e:
                antwoord = f"Er is een fout opgetreden bij het genereren van het antwoord met OpenAI: {e}"
                logging.error(f"Error during OpenAI chat: {e}", exc_info=True)

        print("\n🟢 Antwoord:\n")
        print(antwoord if antwoord else "Geen passend antwoord gevonden.")

        if sources:
            print("\n📚 Bronnen:")
            for source in sorted(list(sources)):
                print(f"- {source}")
        
        if prioritized_pdf_url:
            print(f"\n✨ Aanbevolen PDF: {prioritized_pdf_url}")

# -------------------- Entry Point --------------------
if __name__ == "__main__":
    import asyncio
    asyncio.run(main_rag_chain_loop())