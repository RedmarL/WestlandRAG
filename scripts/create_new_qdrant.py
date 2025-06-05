from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)
collection_name = "westland-mpnet"
vector_size = 768

if client.collection_exists(collection_name):
    print(f"Verwijder bestaande collectie: {collection_name}")
    client.delete_collection(collection_name)

print(f"Maak nieuwe collectie: {collection_name}")
client.create_collection(
    collection_name=collection_name,
    vectors_config={"size": vector_size, "distance": "Cosine"}
)
