from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from keybert import KeyBERT
import ollama
import language_tool_python
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
kw_model = KeyBERT(model)
client = QdrantClient("localhost", port=6333)
tool = language_tool_python.LanguageTool('nl-NL')

# Laad url-beschrijvingen
with open('data/url_descriptions.json', encoding='utf-8') as f:
    url_beschrijvingen = json.load(f)
beschikbare_urls = [
    {"url": url, "beschrijving": beschrijving}
    for url, beschrijving in url_beschrijvingen.items()
]

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

class VraagInput(BaseModel):
    vraag: str

@app.post("/chat")
def chat(vraag_input: VraagInput):
    vraag = vraag_input.vraag.strip()

    # 1. Laat LLM agent de best passende url kiezen
    onderwerpen_lijst = "\n".join(
        [f"{i+1}. {u['beschrijving']} ({u['url']})" for i, u in enumerate(beschikbare_urls)]
    )
    agent_prompt = (
        "Je bent een intelligente informatieagent voor gemeente Westland.\n"
        "Kies welk onderwerp of welke pagina hieronder het meest waarschijnlijk een antwoord geeft op de gebruikersvraag.\n"
        "Antwoord alleen met het nummer van de pagina.\n\n"
        f"Gebruikersvraag: {vraag}\n\n"
        "Beschikbare onderwerpen/pagina's:\n"
        f"{onderwerpen_lijst}"
    )
    agent_response = ollama.chat(
        model='mistral',
        messages=[{"role": "user", "content": agent_prompt}]
    )
    try:
        gekozen_nummer = int(agent_response['message']['content'].strip().split('.')[0]) - 1
    except Exception:
        gekozen_nummer = 0  # fallback naar eerste url als LLM niet juist antwoordt
    # Maak lijst met fallback-volgorde (best -> slechtst)
    volgorde = list(range(len(beschikbare_urls)))
    if gekozen_nummer in volgorde:
        volgorde.remove(gekozen_nummer)
        volgorde = [gekozen_nummer] + volgorde

    # 2. Loop per url: probeer LLM-antwoord op chunks van die url
    for idx in volgorde:
        gekozen_url = beschikbare_urls[idx]['url']
        # Zoek ALLE chunks in Qdrant met deze url
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
        # Qdrant returns tuple (result, next_offset) of alleen result
        chunks = chunks_response[0] if isinstance(chunks_response, tuple) else chunks_response
        if not chunks:
            continue

        context = "\n\n".join([extract_text(c) for c in chunks])
        bronnen_tekst = f"\n\nBronnen:\n[1] {bronvermelding_from_source(gekozen_url)}"
        prompt = (
            "Beantwoord de onderstaande vraag uitsluitend op basis van de context. "
            "Als het antwoord niet letterlijk in de context staat, zeg: 'Het antwoord op uw vraag is niet terug te vinden in de gevonden informatie.'\n\n"
            f"{context}\n\nVraag: {vraag}"
        )

        response = ollama.chat(
            model='mistral',
            messages=[
                {"role": "system", "content": "Geef altijd helder, correct, en natuurlijk Nederlands als antwoord."},
                {"role": "user", "content": prompt}
            ]
        )

        raw_answer = response['message']['content']
        main_part, sep, bronnen_part = raw_answer.partition('\n\nBronnen:\n')
        antwoord_gecorrigeerd = tool.correct(main_part)
        antwoord = antwoord_gecorrigeerd + (sep + bronnen_part if bronnen_part else bronnen_tekst)

        if "Het antwoord op uw vraag is niet terug te vinden" not in antwoord_gecorrigeerd:
            return {"antwoord": antwoord}

    # Niets gevonden in alle urls
    return {"antwoord": "Er is geen specifiek antwoord gevonden op uw vraag in de beschikbare informatie."}
