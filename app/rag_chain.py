from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import ollama
import json
import numpy as np

def bronvermelding_from_source(source_value):
    return source_value.strip() if source_value else "https://www.gemeentewestland.nl/"

def extract_text(hit):
    return hit.payload.get('text', '')

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    # ======== Laden van modellen en data =========
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    client = QdrantClient("localhost", port=6333)

    with open('data/url_descriptions.json', encoding='utf-8') as f:
        url_beschrijvingen = json.load(f)
    beschikbare_urls = [
        {"url": url, "beschrijving": beschrijving}
        for url, beschrijving in url_beschrijvingen.items()
    ]

    with open('data/url_vectors.json', encoding='utf-8') as f:
        url_vectors = json.load(f)

    vraag = "Hoe vraag ik een paspoort aan?"
    vraag_embedding = model.encode(vraag).tolist()

    # ======== Vind top-2 urls op basis van vector similarity ========
    scores = []
    for url_obj in beschikbare_urls:
        url = url_obj["url"]
        emb = url_vectors.get(url)
        if emb:
            score = cosine_similarity(vraag_embedding, emb)
            scores.append((score, url))
    scores.sort(reverse=True)

    # Optioneel: minimaal similarity-eis
    top_urls = [u for s, u in scores[:2] if s > 0.6]  # pas threshold aan indien gewenst

    antwoord = None
    gekozen_url = None

    for gekozen_url in top_urls:
        url_filter = models.Filter(
            must=[models.FieldCondition(key="source", match=models.MatchValue(value=gekozen_url))]
        )
        chunks_response = client.scroll(
            collection_name="westland-mpnet",
            scroll_filter=url_filter,
            with_payload=True
        )
        chunks = chunks_response[0] if isinstance(chunks_response, tuple) else chunks_response
        combined_text = "\n\n".join([extract_text(c) for c in chunks])

        prompt = (
            f"Beantwoord de onderstaande vraag uitsluitend op basis van de context. "
            f"Als het antwoord niet letterlijk in de context staat, zeg: 'Het antwoord op uw vraag is niet terug te vinden in de gevonden informatie.'\n\n"
            f"Gebruik taalgebruik dat overeenkomt met de stijl van de bron.\n\n"
            f"Maak geen spelfouten en gebruik correcte grammatica.\n\n"
            f"Gebruik altijd u in plaats van jij.\n\n"
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

    print("\n🟢 Antwoord:\n")
    print(antwoord if antwoord else "Geen passend antwoord gevonden.")
    print("\n📚 Bron:")
    print(f"[1] {bronvermelding_from_source(gekozen_url) if gekozen_url else 'Geen bron gevonden.'}")

if __name__ == '__main__':
    main()
