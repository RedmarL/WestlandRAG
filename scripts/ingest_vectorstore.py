from qdrant_client import QdrantClient
import pathlib
import pickle
from tqdm import tqdm

# 🔌 Verbinden met Qdrant
client = QdrantClient("localhost", port=6333)
COLLECTIE_NAAM = "westland"

# 🧹 Oude collectie verwijderen
if client.collection_exists(collection_name=COLLECTIE_NAAM):
    client.delete_collection(collection_name=COLLECTIE_NAAM)
    print(f"🧹 Oude collectie '{COLLECTIE_NAAM}' verwijderd.")

# 📦 Nieuwe collectie aanmaken
client.create_collection(
    collection_name=COLLECTIE_NAAM,
    vectors_config={"size": 384, "distance": "Cosine"}
)
print(f"📁 Nieuwe collectie '{COLLECTIE_NAAM}' aangemaakt.")

# 📂 Vectoren laden
EMB_DIR = pathlib.Path('data/embeddings')
points = []

for i, pklfile in enumerate(EMB_DIR.glob('*.pkl')):
    with open(pklfile, 'rb') as f:
        obj = pickle.load(f)
        source = obj.get('source', '')
        if not source.startswith("http"):
            source = "https://www.gemeentewestland.nl/"  # fallback
        points.append({
            "id": i,
            "vector": obj['embedding'],
            "payload": {
                "text": obj['text'],
                "source": source
            }
        })

# ⬆️ Upload in batches
BATCH_GROOTTE = 500
for i in tqdm(range(0, len(points), BATCH_GROOTTE), desc="Upserten"):
    batch = points[i:i + BATCH_GROOTTE]
    client.upsert(collection_name=COLLECTIE_NAAM, points=batch)

print(f"✅ Ingesloten: {len(points)} vectoren in collectie '{COLLECTIE_NAAM}'")

