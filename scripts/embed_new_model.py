from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import json

# Laad je chunks
with open("data/chunks/all_chunks.json", encoding="utf-8") as f:
    all_chunks = json.load(f)

# Init clients en modellen
client = QdrantClient("localhost", port=6333)
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
kw_model = KeyBERT(model)

collection_name = "westland-mpnet"

# === Verwijder collectie als hij bestaat ===
collections = client.get_collections().collections
if any(c.name == collection_name for c in collections):
    print(f"❗ Collectie '{collection_name}' bestaat al. Verwijderen...")
    client.delete_collection(collection_name=collection_name)

# === Maak collectie opnieuw aan ===
print(f"✅ Nieuwe collectie '{collection_name}' wordt aangemaakt...")
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=model.get_sentence_embedding_dimension(),
        distance=models.Distance.COSINE
    )
)

# === Voeg alle chunks toe ===
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
                "keywords": keywords
            }
        })
    client.upsert(
        collection_name=collection_name,
        points=points
    )

print("🎉 Alles is succesvol geüpload naar Qdrant.")

