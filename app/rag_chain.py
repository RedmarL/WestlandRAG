from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import ollama

# 🔍 Init model + vector DB
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
client = QdrantClient("localhost", port=6333)

# ❓ Vraag instellen
vraag = "Hoe vraag ik een paspoort aan?"
vraag_vec = model.encode([vraag])[0].tolist()

# 🔎 Zoek relevante chunks
hits = client.query_points(
    collection_name="westland",
    query=vraag_vec,
    limit=3,
    with_payload=True
)

# 🔗 Gebruik directe URL in source
def bronvermelding_from_source(source_value):
    return source_value.strip() if source_value else "https://www.gemeentewestland.nl/"

# 📦 Haal tekst uit payload
def extract_text(hit):
    if hasattr(hit, 'payload'):
        return hit.payload.get('text', '')
    elif isinstance(hit, dict):
        return hit.get('payload', {}).get('text', '')
    elif isinstance(hit, tuple) and len(hit) >= 3:
        return hit[2].get('text', '')
    return ""

# 📚 Verzamel context en bronvermeldingen
context = "\n\n".join([extract_text(h) for h in hits])

bronnen = set()
for h in hits:
    try:
        if hasattr(h, 'payload'):
            source = h.payload.get('source', '')
        elif isinstance(h, dict):
            source = h.get('payload', {}).get('source', '')
        elif isinstance(h, tuple) and len(h) >= 3:
            source = h[2].get('source', '')
        else:
            continue

        url = bronvermelding_from_source(source)
        bronnen.add(url)
    except Exception:
        continue

# 💬 Prompt voor Ollama
prompt = f"Beantwoord deze vraag op basis van de context:\n{context}\n\nVraag: {vraag}"

response = ollama.chat(
    model='mistral',
    messages=[
        {"role": "system", "content": "Beantwoord alles in het Nederlands."},
        {"role": "user", "content": prompt}
    ]
)

# ✅ Print antwoord + bronnen
print("\n🟢 Antwoord:\n")
print(response['message']['content'])

print("\n📚 Bronvermelding:")
for url in sorted(bronnen):
    print(f"- {url}")
