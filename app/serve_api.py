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
    Bijvoorbeeld:
    '_aanvragen-en-regelen_afval-en-huisvuil_afval_chunk11'
    → 'https://www.gemeentewestland.nl/aanvragen-en-regelen/afval-en-huisvuil/afval'
    """
    # Verwijder '_chunkXX' aan het eind
    parts = source_value.split('_chunk')[0]
    
    # Verwijder leading underscore en vervang underscores door slashes
    clean_path = parts.lstrip('_').replace('_', '/')
    
    # Bouw de URL
    return f"https://www.gemeentewestland.nl/{clean_path}"


def extract_text(hit):
    if hasattr(hit, 'payload'):
        return hit.payload['text']
    elif isinstance(hit, dict) and 'payload' in hit:
        return hit['payload']['text']
    elif isinstance(hit, tuple) and len(hit) >= 3:
        return hit[2]['text']
    else:
        return str(hit)

def extract_source(hit):
    if hasattr(hit, 'payload'):
        return hit.payload.get('source', '')
    elif isinstance(hit, dict) and 'payload' in hit:
        return hit['payload'].get('source', '')
    elif isinstance(hit, tuple) and len(hit) >= 3:
        return hit[2].get('source', '')
    else:
        return ''

# ======= API DEFINITIE =======
class VraagInput(BaseModel):
    vraag: str

@app.post("/chat")
def chat(vraag_input: VraagInput):
    vraag = vraag_input.vraag
    vraag_vec = model.encode([vraag])[0].tolist()

    hits = client.query_points(
        collection_name="westland",
        query=vraag_vec,
        limit=3,
        with_payload=True
    )

    context = "\n\n".join([extract_text(h) for h in hits])
    bronnen = [bronvermelding_from_source(extract_source(h)) for h in hits]
    bronnen_tekst = "\n\nBronnen:\n" + "\n".join(bronnen)

    prompt = f"Beantwoord deze vraag op basis van de context:\n{context}\n\nVraag: {vraag}"

    response = ollama.chat(
        model='mistral',
        messages=[
            {"role": "system", "content": "Beantwoord alles in het Nederlands."},
            {"role": "user", "content": prompt}
        ]
    )

    antwoord = response['message']['content'] + bronnen_tekst
    return {"antwoord": antwoord}
