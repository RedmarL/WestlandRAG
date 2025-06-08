from qdrant_client import QdrantClient
from qdrant_client.http import models

# Start de client
client = QdrantClient("localhost", port=6333)

# Specifieke URL
target_url = "https://www.gemeentewestland.nl/aanvragen-en-regelen/paspoort-id-kaart-en-rijbewijs/paspoort"

# Scroll met filter op de source-URL
filter = models.Filter(
    must=[models.FieldCondition(key="source", match=models.MatchValue(value=target_url))]
)

hits, _ = client.scroll(
    collection_name="westland-openai-embedding",
    scroll_filter=filter,
    with_payload=True,
    limit=20  # eventueel verhogen
)

# Resultaten inspecteren
if not hits:
    print("❌ Geen chunks gevonden voor de opgegeven URL.")
else:
    for i, hit in enumerate(hits, 1):
        print(f"🔹 Chunk {i}")
        print("📦 Payload keys:", hit.payload.keys())
        print("🔑 Keywords:", hit.payload.get("keywords", []))
        print("📝 Tekstsnippet:", hit.payload.get("text", "")[:200].replace("\n", " "))
        print("-" * 60)

