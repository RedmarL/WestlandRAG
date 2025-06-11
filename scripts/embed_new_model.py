#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ingest chunks → OpenAI embeddings → Qdrant, with Dutch keywords per chunk.
Also builds a de-duplicated keyword set for later API use.
"""

import os
import json
import time
import traceback
import pathlib # NEW: For path operations

from openai import OpenAI
from keybert import KeyBERT
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from stop_words import get_stop_words
import tiktoken # For token counting (though not directly used in embedding here, good for context)

# ────────────────────────────── helpers ──────────────────────────────
def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# ─────────────────────────────── config ──────────────────────────────
DATA_PATH        = "data/chunks/all_chunks.json"
KEYWORD_OUTPATH  = "data/keyword_set.json"
COLLECTION_NAME  = "westland-openai-embedding"
EMBED_MODEL      = "text-embedding-3-large" # OpenAI embedding model
EMBED_DIM        = 1536 # Dimension for text-embedding-3-large
KW_MODEL_NAME    = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" # For KeyBERT
BATCH_SIZE       = 128
TOP_KW           = 5
DUTCH_STOP       = get_stop_words("dutch")

# Ensure the output directory for keywords exists
pathlib.Path(KEYWORD_OUTPATH).parent.mkdir(parents=True, exist_ok=True)

# ──────────────────────────── load data ──────────────────────────────
log(f"Loading chunks from {DATA_PATH} …")
try:
    with open(DATA_PATH, encoding="utf-8") as f:
        all_chunks = json.load(f)
    log(f"Loaded {len(all_chunks)} chunks.")
except FileNotFoundError:
    log(f"Error: {DATA_PATH} not found. Please ensure 'allchunks.py' has been run.")
    exit(1)
except json.JSONDecodeError:
    log(f"Error: Could not decode JSON from {DATA_PATH}. Check file integrity.")
    exit(1)

# ─────────────────────────── init clients ────────────────────────────
log("Initializing OpenAI client…")
try:
    client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # Test OpenAI API key (optional but good for early failure)
    # client_openai.models.list()
except Exception as e:
    log(f"Error initializing OpenAI client. Check OPENAI_API_KEY environment variable. Error: {e}")
    exit(1)


log("Initializing Qdrant client…")
qdrant = QdrantClient("localhost", port=6333)

log("Initializing KeyBERT model…")
kw_model = KeyBERT(KW_MODEL_NAME)

# ─────────────────────────── Qdrant setup ────────────────────────────
log(f"Checking Qdrant collection '{COLLECTION_NAME}'…")
try:
    # This will recreate the collection, wiping existing data.
    # For incremental updates, you would use client.get_collection and client.upsert.
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE),
    )
    log(f"Collection '{COLLECTION_NAME}' recreated successfully.")
except Exception as e:
    log(f"Error recreating Qdrant collection: {e}")
    exit(1)

# ─────────────────────────── ingest chunks ───────────────────────────
log("Starting chunk ingestion…")
total_uploaded = 0
keyword_set = set()

# Process chunks in batches
for i in tqdm(range(0, len(all_chunks), BATCH_SIZE), desc="Embedding chunks"):
    batch = all_chunks[i : i + BATCH_SIZE]
    texts_to_embed = [chunk["text"] for chunk in batch]

    t_embed_start = time.time()
    try:
        # Get embeddings from OpenAI
        response = client_openai.embeddings.create(input=texts_to_embed, model=EMBED_MODEL)
        embeddings = [item.embedding for item in response.data]
    except Exception as e:
        log(f"OpenAI embedding failed for batch {i} to {i + BATCH_SIZE}. Error: {e}")
        # Log and skip this batch, or implement retry logic
        continue
    log(f"Batch embedded in {time.time() - t_embed_start:.2f}s")

    points = []
    t_kw_start = time.time()
    for j, chunk in enumerate(batch):
        text = chunk["text"]
        vec = embeddings[j]

        # Extract keywords using KeyBERT
        try:
            kws = [
                kw for kw, _ in kw_model.extract_keywords(
                    text,
                    stop_words=DUTCH_STOP,
                    top_n=TOP_KW,
                    use_mmr=True,
                    diversity=0.7,
                )
            ]
        except Exception as e:
            log(f"Keyword extraction failed on chunk {chunk.get('id', 'N/A')}: {e}")
            traceback.print_exc()
            kws = []

        keyword_set.update(kws)

        if j == 0: # Log sample chunk from first batch
            log(f"Sample chunk id={chunk.get('id', 'N/A')} → kws={kws}")

        points.append(
            models.PointStruct(
                id=str(chunk["id"]), # Use the UUID generated in chunk.py as the Qdrant ID
                vector=vec,
                payload={
                    "text": text,
                    "source": chunk.get("source", "unknown"), # Use .get for robustness
                    "keywords": kws,
                    "document_type": chunk.get("document_type", "unknown"),
                    "document_title": chunk.get("document_title", "Geen Titel"),
                    "department": chunk.get("department", "Algemeen"),
                    "last_modified_date": chunk.get("last_modified_date", ""),
                    "original_file_name": chunk.get("original_file_name", ""),
                    "chunk_idx": chunk.get("chunk_idx", 0) # Include chunk index for ordering within doc if needed
                },
            )
        )

    log(f"Keywords for {len(batch)} chunks extracted in {time.time() - t_kw_start:.2f}s")

    t_upsert = time.time()
    try:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        log(f"Upserted {len(points)} points in {time.time() - t_upsert:.2f}s")
        total_uploaded += len(points)
    except Exception as e:
        log(f"Error during Qdrant upsert for batch {i} to {i + BATCH_SIZE}. Error: {e}")
        traceback.print_exc()

# ─────────────────────────── after ingest ────────────────────────────
log(f"\n🎉 Done! {total_uploaded} chunks embedded and stored in Qdrant.")
log(f"Saving {len(keyword_set)} unique keywords → {KEYWORD_OUTPATH}")
try:
    with open(KEYWORD_OUTPATH, "w", encoding="utf-8") as out:
        json.dump(list(keyword_set), out, ensure_ascii=False, indent=2)
    log("Keyword set saved.")
except Exception as e:
    log(f"Error saving keyword set: {e}")