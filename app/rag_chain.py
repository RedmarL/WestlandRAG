from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import ollama
import json
import numpy as np

# ======== Laden van modellen en data =========
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
client = QdrantClient("localhost", port=6333)

# Laad de url-beschrijvingen en url-vectors
with open('data/url_descriptions.json', encoding='utf-8') as f:
    url_beschrijvingen = json.load(f)
beschikbare_urls = [
    {"url": url, "beschrijving": beschrijving}
    for url, beschrijving in url_beschrijvingen.items()
]

with open('data/url_vectors.json', encoding='utf-8') as f:
    url_vectors = json.load(f)

def bronvermelding_from_source(source_value):
    return source_value.strip() if source_value else "https://www.gemeentewestland.nl/"

def extract_text(hit):
    return hit.payload.get('text', '')

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ======== De gebruikersvraag =========
vraag = "Hoe vraag ik een paspoort aan?"
vraag_embedding = model.encode(vraag).tolist()

# ======== Vind top-2 urls op basis van vector similarity ========
scores = []
for url_obj in beschikbare_urls:
    url = url_obj["url"]
    emb = url_vectors[url]
    score = cosine_similarity(vraag_embedding, emb)
    scores.append((score, url))
scores.sort(reverse=True)  # hoogste similarity eerst

# Pak max 2 urls (beste eerst)
top_urls = [u for _, u in scores[:2]]

antwoord = None
gekozen_url = None

for gekozen_url in top_urls:
    url_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="source",
                match=models.MatchValue(value=gekozen_url)
            )
        ]
    )
    chunks_response = client.scroll(
        collection_name="westland-mpnet",
        scroll_filter=url_filter,
        with_payload=True
    )
    chunks = chunks_response[0] if isinstance(chunks_response, tuple) else chunks_response

    # Voeg de tekst van alle chunks samen
    combined_text = "\n\n".join([extract_text(c) for c in chunks])

    prompt = (
        f"Beantwoord de onderstaande vraag uitsluitend op basis van de context. "
        f"Als het antwoord niet letterlijk in de context staat, zeg: 'Het antwoord op uw vraag is niet terug te vinden in de gevonden informatie.'\n\n"
        f"Gebruik taalgebruik dat overeenkomt met de stijl van de bron.\n\n"
        f"Maak geen spelfouten en gebruik correcte grammatica.\n\n"
        f"{combined_text}\n\nVraag: {vraag}"
    )

    response = ollama.chat(
        model='mistral',
        messages=[
            {"role": "system", "content": "Beantwoord alles in het Nederlands."},
            {"role": "user", "content": prompt}
        ]
    )

    antwoord = response['message']['content']
    if "Het antwoord op uw vraag is niet terug te vinden" not in antwoord:
        break

# Toon het antwoord en de bron
print("\n🟢 Antwoord:\n")
print(antwoord)
print("\n📚 Bron:")
print(f"[1] {bronvermelding_from_source(gekozen_url)}")
