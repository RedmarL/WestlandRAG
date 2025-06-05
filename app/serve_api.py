from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import ollama

app = FastAPI()

# CORS toestaan (voor localhost frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======= MODEL EN VECTORSTORE =======
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
client = QdrantClient("localhost", port=6333)

# ======= HELPER FUNCTIES =======
def bronvermelding_from_source(source_value):
    """
    Zet een chunknaam om in de juiste URL.
    Als de bron al een volledige url is, geef gewoon die terug.
    """
    if not source_value:
        return "https://www.gemeentewestland.nl/"
    if source_value.startswith("http"):
        return source_value
    parts = source_value.split('_chunk')[0]
    clean_path = parts.lstrip('_').replace('_', '/')
    return f"https://www.gemeentewestland.nl/{clean_path}"

def extract_text(hit):
    return hit.payload.get('text', '')

def extract_source(hit):
    return hit.payload.get('source', '')

# ======= API DEFINITIE =======
class VraagInput(BaseModel):
    vraag: str

@app.post("/chat")
def chat(vraag_input: VraagInput):
    vraag = vraag_input.vraag
    vraag_vec = model.encode([vraag])[0].tolist()

    hits_response = client.query_points(
        collection_name="westland",
        query=vraag_vec,
        limit=3,
        with_payload=True
    )
    hits = hits_response.points

    context = "\n\n".join([extract_text(h) for h in hits])
    # Unieke bronnen, volgorde behouden
    bronnen = []
    for h in hits:
        url = bronvermelding_from_source(extract_source(h))
        if url not in bronnen:
            bronnen.append(url)

    bronnen_tekst = "\n\nBronnen:\n" + "\n".join(f"[{i+1}] {b}" for i, b in enumerate(bronnen))

    prompt = (
        "Beantwoord de onderstaande vraag op basis van de context. "
        "Schrijf een helder en vloeiend antwoord. Let op spelling en grammatica. Gebruik altijd u of uw.\n"
        "Je hoeft in de tekst geen bronnen te noemen; deze worden eronder weergegeven.\n\n"
        f"{context}\n\nVraag: {vraag}"
    )

    response = ollama.chat(
        model='mistral',
        messages=[
            {"role": "system", "content": "Beantwoord alles in het Nederlands."},
            {"role": "user", "content": prompt}
        ]
    )

    antwoord = response['message']['content'] + bronnen_tekst
    return {"antwoord": antwoord}
