from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

import json

# Laad je chunks
with open("data/chunks/all_chunks.json", encoding="utf-8") as f:
    all_chunks = json.load(f)

client = QdrantClient("localhost", port=6333)
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
kw_model = KeyBERT(model)

batch_size = 128
for i in range(0, len(all_chunks), batch_size):
    batch = all_chunks[i:i+batch_size]
    vectors = model.encode([c["text"] for c in batch], show_progress_bar=True)
    points = []
    for j, chunk in enumerate(batch):
        keywords = [kw for kw, _ in kw_model.extract_keywords(chunk['text'], stop_words='dutch', top_n=3)]
        points.append({
            "id": i + j,
            "vector": vectors[j].tolist(),
            "payload": {
                "text": chunk["text"],
                "source": chunk["source"],
                "keywords": keywords  # <<<< HIER!
            }
        })
    client.upsert(
        collection_name="westland-mpnet",
        points=points
    )
