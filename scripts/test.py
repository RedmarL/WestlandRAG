from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)
hits = client.scroll(collection_name="westland", limit=5, with_payload=True)

for hit in hits[0]:
    print(hit.payload)
