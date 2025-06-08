#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ingest chunks → OpenAI embeddings → Qdrant, with Dutch keywords per chunk.
Also builds a de-duplicated keyword set for later API use.
"""

import os, json, time, traceback
from openai import OpenAI
from keybert import KeyBERT
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from stop_words import get_stop_words
import tiktoken

# ────────────────────────────── helpers ──────────────────────────────
def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# ─────────────────────────────── config ──────────────────────────────
DATA_PATH        = "data/chunks/all_chunks.json"
KEYWORD_OUTPATH  = "data/keyword_set.json"      # ← NEW
COLLECTION_NAME  = "westland-openai-embedding"
EMBED_MODEL      = "text-embedding-3-large"
EMBED_DIM        = 1536
KW_MODEL_NAME    = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
BATCH_SIZE       = 128
TOP_KW           = 5
DUTCH_STOP       = get_stop_words("dutch")

# ──────────────────────────── load data ──────────────────────────────
log(f"Loading chunks from {DATA_PATH} …")
with open(DATA_PATH, encoding="utf-8") as f:
    all_chunks = json.load(f)
log(f"Loaded {len(all_chunks)} chunks.")

# ───────────────────────── initialise clients ────────────────────────
log("Initialising clients and models …")
qdrant        = QdrantClient(host="localhost", port=6333)
openai_client = OpenAI()
kw_model      = KeyBERT(SentenceTransformer(KW_MODEL_NAME))

# ────────────────────────── collection setup ─────────────────────────
if qdrant.collection_exists(collection_name=COLLECTION_NAME):
    log(f"❗ Collection '{COLLECTION_NAME}' already exists – deleting.")
    qdrant.delete_collection(collection_name=COLLECTION_NAME)

log(f"✅ Creating collection '{COLLECTION_NAME}' …")
qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE),
)

# ─────────────────────────── embed helper ────────────────────────────
def truncate_text(text, max_tokens=8000, model_name=EMBED_MODEL):
    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(text)
    return enc.decode(tokens[:max_tokens]) if len(tokens) > max_tokens else text

def get_openai_embeddings(texts, model=EMBED_MODEL):
    clean = [truncate_text(t.replace("\n", " ")) for t in texts]
    resp  = openai_client.embeddings.create(input=clean, model=model, dimensions=EMBED_DIM)
    return [d.embedding for d in resp.data]

# ───────────────────────────── main loop ─────────────────────────────
total_uploaded  = 0
keyword_set: set[str] = set()                       # ← NEW
batches         = range(0, len(all_chunks), BATCH_SIZE)
log(f"Starting upload: {len(batches)} batches of ≤{BATCH_SIZE} chunks each.")

for batch_idx, start in enumerate(batches, 1):
    log(f"\n— Batch {batch_idx}/{len(batches)} —")
    batch   = all_chunks[start:start + BATCH_SIZE]
    texts   = [c["text"] for c in batch]
    vectors = get_openai_embeddings(texts)
    log(f"Embedding vector length: {len(vectors[0])}")

    points = []
    t_kw_start = time.time()

    for j, (chunk, vec) in enumerate(zip(batch, vectors)):
        idx, text = start + j, chunk["text"]

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
            log(f"Keyword extraction failed on chunk {idx}: {e}")
            traceback.print_exc()
            kws = []

        keyword_set.update(kws)                     # ← NEW

        if j == 0:
            log(f"Sample chunk id={idx} → kws={kws}")

        points.append(
            models.PointStruct(
                id=idx,
                vector=vec,
                payload={"text": text, "source": chunk["source"], "keywords": kws},
            )
        )

    log(f"Keywords for {len(batch)} chunks extracted in {time.time() - t_kw_start:.2f}s")

    t_upsert = time.time()
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    log(f"Upserted {len(points)} points in {time.time() - t_upsert:.2f}s")
    total_uploaded += len(points)

# ─────────────────────────── after ingest ────────────────────────────
log(f"\n🎉 Done! {total_uploaded} chunks embedded and stored in Qdrant.")
log(f"Saving {len(keyword_set)} unique keywords → {KEYWORD_OUTPATH}")
os.makedirs(os.path.dirname(KEYWORD_OUTPATH), exist_ok=True)
with open(KEYWORD_OUTPATH, "w", encoding="utf-8") as f:
    json.dump(sorted(keyword_set), f, ensure_ascii=False, indent=2)
log("Keyword set saved successfully.")
