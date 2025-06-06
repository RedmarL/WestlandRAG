from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import json

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
client = QdrantClient("localhost", port=6333)

with open("data/url_descriptions.json", encoding="utf-8") as f:
    url_data = json.load(f)  # Dict[str, str]

urls = list(url_data.keys())
descriptions = list(url_data.values())

vectors = model.encode(descriptions, show_progress_bar=True)

points = []
for i, (url, desc) in enumerate(zip(urls, descriptions)):
    points.append({
        "id": i,
        "vector": vectors[i].tolist(),
        "payload": {
            "url": url,
            "description": desc
        }
    })

client.recreate_collection(
    collection_name="westland-url-descriptions",
    vectors_config={"size": len(vectors[0]), "distance": "Cosine"}
)

client.upsert(collection_name="westland-url-descriptions", points=points)
