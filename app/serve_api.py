#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import ast
import re
from time import time
import traceback
import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import concurrent.futures # For parallel Qdrant searches
from keybert import KeyBERT # For query keyword extraction
from stop_words import get_stop_words # For KeyBert stop words

# --- Configuration ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check if the API key is loaded
if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY environment variable not set. Please set it before running the app.")
    # Exit early if a critical dependency is missing at import time
    exit(1)

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_HTML = "westland-openai-embedding"
QDRANT_COLLECTION_PDF_LINKS = "westland-pdf-links"
KEYWORD_PATH = "data/keyword_set.json"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- FastAPI App Setup ---
app = FastAPI(title="Gemeente Westland RAG API", description="API voor het beantwoorden van vragen over de Gemeente Westland.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global instances (initialized on startup) ---
openai_client: OpenAI = None
qdrant_client: QdrantClient = None
embedding_model: SentenceTransformer = None
kw_model: KeyBERT = None # KeyBERT model for query keyword extraction
KEYWORDS: set = set() # Global keyword set from HTML chunks

# --- Pydantic Models for Request/Response ---
class QueryRequest(BaseModel):
    query: str
    top_k_html: int = 5
    top_k_pdf: int = 3
    min_score_html: float = 0.5
    min_score_pdf: float = 0.45

class Source(BaseModel):
    url: str
    document_type: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    score: Optional[float] = None

class RagResponse(BaseModel):
    answer: str
    sources: List[Source]
    prioritized_pdf_link: Optional[str] = None

# --- Startup Event to load models and clients ---
@app.on_event("startup")
async def startup_event():
    global openai_client, qdrant_client, embedding_model, kw_model, KEYWORDS

    logging.info("Starting up RAG API...")

    # 1. Initialize OpenAI Client
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        openai_client.models.list() # Test connection
        logging.info("OpenAI client initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client or connect: {e}", exc_info=True)
        raise RuntimeError("Failed to connect to OpenAI API. Check OPENAI_API_KEY.")

    # 2. Initialize Qdrant Client
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        # Verify collections exist
        qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_HTML)
        qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_PDF_LINKS)
        logging.info("Qdrant client initialized and collections verified.")
    except Exception as e:
        logging.error(f"Failed to connect to Qdrant or collections not found: {e}", exc_info=True)
        raise RuntimeError("Failed to connect to Qdrant or required collections missing.")

    # 3. Load Embedding Model for queries
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logging.info(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load SentenceTransformer model '{EMBEDDING_MODEL_NAME}': {e}", exc_info=True)
        raise RuntimeError("Failed to load embedding model.")

    # 4. Load KeyBERT model for query keyword extraction
    try:
        kw_model = KeyBERT(EMBEDDING_MODEL_NAME) # Using the same model as for embeddings
        logging.info(f"KeyBERT model initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to load KeyBERT model: {e}", exc_info=True)
        # Not a critical error if keyword boosting is optional
        kw_model = None

    # 5. Load Keywords from file
    try:
        with open(KEYWORD_PATH, encoding="utf-8") as f:
            KEYWORDS = set(json.load(f))
        logging.info(f"Loaded {len(KEYWORDS)} keywords from {KEYWORD_PATH}.")
    except FileNotFoundError:
        logging.warning(f"Keyword file not found at {KEYWORD_PATH}. Keyword filtering/boosting might be less effective.")
    except json.JSONDecodeError:
        logging.error(f"Error decoding keywords from {KEYWORD_PATH}. File might be corrupted.")

# --- Utility Functions ---
def nesting_level(url: str) -> int:
    """Calculates the nesting level of a URL path."""
    path = url.split("://", 1)[-1].split("/", 1)[-1]
    return path.count("/")

def hoofdsegment(url: str) -> str:
    """Extracts the main segment of a URL path (e.g., 'aanvragen-en-regelen')."""
    parts = url.split("://", 1)[-1].split("/", 1)
    if len(parts) > 1:
        return parts[1].split("/")[0]
    return ""

# --- Core Retrieval Logic ---
async def retrieve_and_rank_documents(
    query: str,
    top_k_html: int,
    top_k_pdf: int,
    min_score_html: float,
    min_score_pdf: float
) -> Tuple[List[Dict], List[Dict], Optional[Dict]]:
    """
    Retrieves relevant HTML chunks and PDF links from Qdrant,
    and identifies the top prioritized PDF link if applicable.
    """
    query_vector = embedding_model.encode(query).tolist()
    
    query_keywords = set()
    if kw_model: # Only try to extract keywords if KeyBERT model loaded successfully
        try:
            query_keywords = set(kw for kw, _ in kw_model.extract_keywords(
                query,
                stop_words=get_stop_words("dutch"),
                top_n=3,
                use_mmr=True,
                diversity=0.7,
            ))
        except Exception as e:
            logging.warning(f"Failed to extract keywords for query: {e}")

    html_hits = []
    pdf_link_hits = []

    # Using ThreadPoolExecutor for concurrent blocking I/O calls to Qdrant
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_html = executor.submit(
            qdrant_client.search,
            collection_name=QDRANT_COLLECTION_HTML,
            query_vector=query_vector,
            limit=top_k_html,
            score_threshold=min_score_html,
            with_payload=True,
            with_vectors=False
        )
        future_pdf = executor.submit(
            qdrant_client.search,
            collection_name=QDRANT_COLLECTION_PDF_LINKS,
            query_vector=query_vector,
            limit=top_k_pdf,
            score_threshold=min_score_pdf,
            with_payload=True,
            with_vectors=False
        )

        try:
            html_hits = future_html.result()
            logging.info(f"Retrieved {len(html_hits)} HTML chunks.")
        except Exception as e:
            logging.error(f"Error searching HTML collection: {e}", exc_info=True)
        
        try:
            pdf_link_hits = future_pdf.result()
            logging.info(f"Retrieved {len(pdf_link_hits)} PDF links.")
        except Exception as e:
            logging.error(f"Error searching PDF links collection: {e}", exc_info=True)

    relevant_html_chunks = []
    for hit in html_hits:
        chunk_kws = set(hit.payload.get("keywords", []))
        overlap = query_keywords & chunk_kws
        relevance_score = hit.score
        if overlap:
            relevance_score += len(overlap) * 0.05 # Small boost for keyword overlap

        relevant_html_chunks.append({
            "score": relevance_score,
            "text": hit.payload.get("text", ""),
            "source": hit.payload.get("source", "unknown"),
            "document_type": hit.payload.get("document_type", "html_page"),
            "document_title": hit.payload.get("document_title", "Geen Titel"),
            "keywords": hit.payload.get("keywords", []),
            "chunk_idx": hit.payload.get("chunk_idx", 0)
        })
    relevant_html_chunks.sort(key=lambda x: x["score"], reverse=True)

    relevant_pdf_links = []
    prioritized_pdf_link = None
    best_pdf_score = 0.0

    for hit in pdf_link_hits:
        pdf_url = hit.payload.get("url")
        pdf_title = hit.payload.get("title", f"PDF Document ({pdf_url.split('/')[-1]})") # Default title if not in payload
        pdf_score = hit.score

        # Prioritization logic for PDF links:
        # If query contains document-specific keywords and PDF score is high
        is_document_query = any(kw in query.lower() for kw in ["notulen", "beleid", "verslag", "vergadering", "jaarverslag", "document", "raadvergadering"])
        
        if is_document_query and pdf_score >= min_score_pdf:
            if pdf_score > best_pdf_score:
                best_pdf_score = pdf_score
                prioritized_pdf_link = {
                    "url": pdf_url,
                    "document_type": "pdf_link",
                    "title": pdf_title,
                    "snippet": None, # No snippet for link, but could be a description if available
                    "score": pdf_score
                }
        
        relevant_pdf_links.append({
            "url": pdf_url,
            "document_type": "pdf_link",
            "title": pdf_title,
            "snippet": None,
            "score": pdf_score
        })
    relevant_pdf_links.sort(key=lambda x: x["score"], reverse=True)

    return relevant_html_chunks, relevant_pdf_links, prioritized_pdf_link

# --- LLM Response Generation ---
async def generate_llm_response(
    user_query: str,
    relevant_html_chunks: List[Dict],
    relevant_pdf_links: List[Dict],
    prioritized_pdf_link: Optional[Dict]
) -> Tuple[str, List[Source], Optional[str]]:
    """
    Generates an LLM response based on retrieved context, prioritizing PDF links.
    """
    sources_for_response: List[Source] = []
    
    # Process HTML chunks for context and sources
    html_content_by_url = defaultdict(lambda: {"title": "Onbekend", "chunks": [], "score": 0.0})
    for chunk in relevant_html_chunks:
        url = chunk["source"]
        html_content_by_url[url]["title"] = chunk["document_title"]
        # Ensure chunks are ordered by chunk_idx to maintain document flow
        html_content_by_url[url]["chunks"].append(chunk)
        html_content_by_url[url]["score"] = max(html_content_by_url[url]["score"], chunk["score"])

    combined_html_context_strings = []
    for url, data in html_content_by_url.items():
        sorted_chunks_for_url = sorted(data["chunks"], key=lambda x: x.get("chunk_idx", 0))
        full_text_for_url = "\n".join([c["text"] for c in sorted_chunks_for_url])
        combined_html_context_strings.append(f"Document Titel: {data['title']}\nBron: {url}\nInhoud:\n{full_text_for_url}\n---")
        
        sources_for_response.append(Source(
            url=url,
            document_type="html_page",
            title=data["title"],
            snippet=full_text_for_url[:300] + "..." if len(full_text_for_url) > 300 else full_text_for_url,
            score=data["score"]
        ))
    
    combined_html_context = "\n\n".join(combined_html_context_strings)

    final_answer = ""
    primary_pdf_url_output = None

    if prioritized_pdf_link:
        pdf_url = prioritized_pdf_link["url"]
        pdf_title = prioritized_pdf_link["title"]
        primary_pdf_url_output = pdf_url

        # Add prioritized PDF to the beginning of sources list
        sources_for_response.insert(0, Source(
            url=pdf_url,
            document_type="pdf_link",
            title=pdf_title,
            snippet=f"Volledig document beschikbaar op deze link.",
            score=prioritized_pdf_link["score"]
        ))

        # LLM prompt for prioritizing PDF link
        system_prompt = (
            "Je bent een behulpzame assistent van de Gemeente Westland. "
            "De gebruiker heeft een vraag gesteld waarvoor een relevant PDF-document is gevonden. "
            "Antwoord de gebruiker kort, direct en beleefd, en verwijs in je antwoord direct naar het gevonden document "
            "door de link te noemen. Gebruik de onderstaande HTML-context alleen als aanvulling, maar de PDF-link is de primaire bron. "
            "Formuleer je antwoord in het Nederlands.\n\n"
            f"De meest relevante informatie is te vinden in het document: {pdf_title}. Directe link: {pdf_url}"
        )
        user_prompt_content = f"Gebruikersvraag: {user_query}\n\nOverige relevante HTML-context:\n{combined_html_context}"
        
        # We explicitly instruct the LLM to give the link first, and then elaborate if possible
        # This part ensures the link is always present even if LLM slightly deviates
        final_answer_prefix = f"Om uw vraag over '{user_query}' te beantwoorden, kunt u het volledige document vinden op: {pdf_url}. "

    else:
        # No strong PDF priority, answer from HTML and list other PDF links as general sources
        system_prompt = (
            "Je bent een behulpzame assistent van de Gemeente Westland. "
            "Beantwoord de gebruikersvraag zo nauwkeurig mogelijk op basis van de onderstaande HTML-context. "
            "Als je geen duidelijk antwoord kunt vinden, geef dan aan dat je het antwoord niet in de beschikbare informatie kunt vinden. "
            "Vermeld de bronnen (URLs) die je gebruikt hebt. Formuleer je antwoord in het Nederlands.\n\n"
            "Beschikbare HTML-context:\n"
            f"{combined_html_context}"
        )
        user_prompt_content = f"Gebruikersvraag: {user_query}"
        final_answer_prefix = ""

        # Add other relevant PDF links to sources if they weren't prioritized
        for pdf_link in relevant_pdf_links:
            sources_for_response.append(Source(
                url=pdf_link["url"],
                document_type="pdf_link",
                title=pdf_link["title"],
                snippet=f"Verwante PDF: {pdf_link['url'].split('/')[-1]}",
                score=pdf_link["score"]
            ))

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_content}
    ]

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o", # Or gpt-3.5-turbo if cost/speed is critical and quality is acceptable
            messages=messages,
            temperature=0.2,
            max_tokens=500
        )
        llm_generated_text = response.choices[0].message.content.strip()
        final_answer = final_answer_prefix + llm_generated_text

        return final_answer, sources_for_response, primary_pdf_url_output

    except Exception as e:
        logging.error(f"Error generating LLM response: {e}", exc_info=True)
        return "Er is een probleem opgetreden bij het genereren van het antwoord.", [], None


# --- FastAPI Endpoint ---
@app.post("/query", response_model=RagResponse)
async def query_rag(request: QueryRequest):
    """
    Handles user queries, performs RAG using both HTML chunks and PDF links,
    and returns a structured response.
    """
    try:
        start_time = time()

        # Step 1: Retrieve relevant documents (HTML chunks and PDF links)
        html_chunks, pdf_links, prioritized_pdf_link = await retrieve_and_rank_documents(
            request.query,
            request.top_k_html,
            request.top_k_pdf,
            request.min_score_html,
            request.min_score_pdf
        )
        
        # Step 2: Generate LLM response based on retrieved context and prioritization
        llm_answer, sources, primary_pdf_url = await generate_llm_response(
            request.query,
            html_chunks,
            pdf_links,
            prioritized_pdf_link
        )

        end_time = time()
        processing_time = end_time - start_time
        logging.info(f"Query '{request.query}' processed in {processing_time:.2f} seconds.")

        return RagResponse(
            answer=llm_answer,
            sources=sources,
            prioritized_pdf_link=primary_pdf_url
        )

    except RuntimeError as e:
        # Catch errors from startup event or critical initialization
        logging.error(f"Critical startup/runtime error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"De server is niet correct geïnitialiseerd of er is een kritieke fout opgetreden: {e}"
        )
    except Exception as e:
        logging.error(f"Unhandled error during query processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Er is een onverwachte fout opgetreden bij het verwerken van uw vraag."
        )