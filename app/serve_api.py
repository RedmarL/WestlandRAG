from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import ollama
import language_tool_python
import json
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
client = QdrantClient("localhost", port=6333)
tool = language_tool_python.LanguageTool('nl-NL')

# Laad url-beschrijvingen en vectoren
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

class VraagInput(BaseModel):
    vraag: str

@app.post("/chat")
def chat(vraag_input: VraagInput):
    vraag = vraag_input.vraag.strip()
    vraag_embedding = model.encode(vraag).tolist()

    # 1. Bereken similarity met alle urls, pak top-2
    scores = []
    for url_obj in beschikbare_urls:
        url = url_obj["url"]
        emb = url_vectors.get(url)
        if emb:
            score = cosine_similarity(vraag_embedding, emb)
            scores.append((score, url))
    scores.sort(reverse=True)
    top_urls = [u for _, u in scores[:2]]

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
        if not chunks:
            continue

        combined_text = "\n\n".join([extract_text(c) for c in chunks])

        # ===== 1e prompt: Genereer rauw, feitelijk antwoord =====
        eerste_prompt = (
            "Geef een direct, feitelijk antwoord op de onderstaande vraag op basis van de context. "
            "Gebruik waar mogelijk letterlijk zinnen uit de context. Voeg geen uitleg of opsmuk toe.\n\n"
            f"Context:\n{combined_text}\n\nVraag: {vraag}"
        )
        eerste_response = ollama.chat(
            model='bramvanroy/geitje-7b-ultra:Q5_K_M',
            messages=[
                {"role": "system", "content": "Beantwoord in feitelijk, correct Nederlands."},
                {"role": "user", "content": eerste_prompt}
            ]
        )
        ruw_antwoord = eerste_response['message']['content']

        # ===== 2e prompt: Reflecteer, structureer en motiveer =====
        tweede_prompt = (
            "Hieronder staat een feitelijk antwoord op een vraag over gemeentelijke dienstverlening.\n"
            "Verbeter dit antwoord door te controleren of het duidelijk, volledig en bruikbaar is voor een burger die deze vraag stelt. "
            "Vul het antwoord aan met relevante details uit de context als dat nodig is om de vraag écht te beantwoorden, maar voeg niets toe wat niet in de context voorkomt.\n\n"
            "Splits het resultaat in een kernboodschap (snippet/hoofdantwoord) en eventuele bijzinnen of waarschuwingen "
            "(zoals zinnen beginnend met 'Let op', 'Bijzonderheden', 'Bent u', etc.).\n"
            "Geef de output in het volgende JSON-formaat:\n"
            "{\n"
            '  "snippet": "<duidelijke hoofdantwoord>",\n'
            '  "extra": ["<bijzonderheid 1>", "<bijzonderheid 2>", ...],\n'
            '  "toelichting": "<korte uitleg waarom je deze selectie en aanvulling hebt gemaakt>"\n'
            "}\n\n"
            f"Context:\n{combined_text}\n\n"
            f"Oorspronkelijk antwoord:\n{ruw_antwoord}"
        )

        tweede_response = ollama.chat(
            model='mistral',
            messages=[
                {"role": "system", "content": "Je bent een behulpzame taalassistent."},
                {"role": "user", "content": tweede_prompt}
            ]
        )

        # Probeer JSON te parsen, anders fallback op ruwe tekst
        try:
            answer_json = json.loads(tweede_response['message']['content'])
            hoofd_correct = tool.correct(answer_json.get("snippet", ""))
            extra_correct = [tool.correct(e) for e in answer_json.get("extra", [])]
            toelichting = tool.correct(answer_json.get("toelichting", ""))
        except Exception:
            hoofd_correct = tool.correct(tweede_response['message']['content'])
            extra_correct = []
            toelichting = ""

        resultaat = {
            "raw": ruw_antwoord,
            "snippet": hoofd_correct,
            "extra": extra_correct,
            "toelichting": toelichting,
            "bron": bronvermelding_from_source(gekozen_url)
        }

        if (
            "Het antwoord op uw vraag is niet terug te vinden" not in hoofd_correct
            and "geen specifiek antwoord gevonden" not in hoofd_correct.lower()
        ):
            return {"resultaat": resultaat}

    # Geen antwoord gevonden
    return {"resultaat": {
        "raw": "",
        "snippet": "Er is geen specifiek antwoord gevonden op uw vraag in de beschikbare informatie.",
        "extra": [],
        "toelichting": "",
        "bron": ""
    }}
