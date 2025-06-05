from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import ollama
import language_tool_python

app = FastAPI()

# CORS toestaan (voor localhost frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======= MODEL EN VECTORSTORE =======
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
client = QdrantClient("localhost", port=6333)

tool = language_tool_python.LanguageTool('nl-NL')

def bronvermelding_from_source(source_value):
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

class VraagInput(BaseModel):
    vraag: str

@app.post("/chat")
def chat(vraag_input: VraagInput):
    vraag = vraag_input.vraag
    vraag_vec = model.encode([vraag])[0].tolist()

    hits_response = client.query_points(
        collection_name="westland-mpnet",
        query=vraag_vec,
        limit=3,
        with_payload=True
    )
    hits = hits_response.points

    context = "\n\n".join([extract_text(h) for h in hits])
    bronnen = []
    for h in hits:
        url = bronvermelding_from_source(extract_source(h))
        if url not in bronnen:
            bronnen.append(url)
    bronnen_tekst = "\n\nBronnen:\n" + "\n".join(f"[{i+1}] {b}" for i, b in enumerate(bronnen))

    # **Prompt: nog duidelijker Nederlands!**
    prompt = (
        "Beantwoord de onderstaande vraag op basis van de context. "
        "Schrijf een helder, vloeiend en correct antwoord in natuurlijk Nederlands, "
        "zonder spelfouten of kromme zinnen. Je hoeft bronnen niet in de tekst te noemen; deze worden eronder weergegeven.\n\n"
        f"{context}\n\nVraag: {vraag}"
    )

    response = ollama.chat(
        model='mistral',
        messages=[
            {"role": "system", "content": "Geef altijd helder, correct, en natuurlijk Nederlands als antwoord."},
            {"role": "user", "content": prompt}
        ]
    )

    # **Spellingscontrole alleen op het hoofdantwoord**
    raw_answer = response['message']['content']
    main_part, sep, bronnen_part = raw_answer.partition('\n\nBronnen:\n')
    antwoord_gecorrigeerd = tool.correct(main_part)
    antwoord = antwoord_gecorrigeerd + (sep + bronnen_part if bronnen_part else bronnen_tekst)
    return {"antwoord": antwoord}
