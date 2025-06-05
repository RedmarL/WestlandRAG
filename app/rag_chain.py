from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import ollama

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
client = QdrantClient("localhost", port=6333)

vraag = "Hoe vraag ik een paspoort aan?"
vraag_vec = model.encode([vraag])[0].tolist()

hits_response = client.query_points(
    collection_name="westland",
    query=vraag_vec,
    limit=3,
    with_payload=True
)
hits = hits_response.points

def bronvermelding_from_source(source_value):
    return source_value.strip() if source_value else "https://www.gemeentewestland.nl/"

def extract_text(hit):
    return hit.payload.get('text', '')

def extract_source(hit):
    return hit.payload.get('source', '')

context = "\n\n".join([extract_text(h) for h in hits])
bronnen = [bronvermelding_from_source(extract_source(h)) for h in hits]
unieke_bronnen = []
[unieke_bronnen.append(b) for b in bronnen if b not in unieke_bronnen]

prompt = (
    f"Beantwoord de onderstaande vraag op basis van de context. "
    f"Geef een helder, vloeiend antwoord. Je hoeft geen bronnen in de tekst te vermelden, maar onder het antwoord worden de geraadpleegde bronnen getoond.\n\n"
    f"{context}\n\nVraag: {vraag}"
)

response = ollama.chat(
    model='mistral',
    messages=[
        {"role": "system", "content": "Beantwoord alles in het Nederlands."},
        {"role": "user", "content": prompt}
    ]
)

print("\n🟢 Antwoord:\n")
print(response['message']['content'])
print("\n📚 Bronnen:")
for i, url in enumerate(unieke_bronnen, 1):
    print(f"[{i}] {url}")
