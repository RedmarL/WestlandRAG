# embed_upload_chunks.py
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import json

client = QdrantClient("localhost", port=6333)
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# Let op het juiste pad!
with open("data/chunks/all_chunks.json", encoding="utf-8") as f:
    all_chunks = json.load(f)

batch_size = 128
for i in range(0, len(all_chunks), batch_size):
    batch = all_chunks[i:i+batch_size]
    vectors = model.encode([c["text"] for c in batch], show_progress_bar=True)
    points = []
    for j, chunk in enumerate(batch):
        points.append({
            "id": i + j,
            "vector": vectors[j].tolist(),
            "payload": {
                "text": chunk["text"],
                "source": chunk["source"]
            }
        })
    client.upsert(
        collection_name="westland-mpnet",
        points=points
    )
