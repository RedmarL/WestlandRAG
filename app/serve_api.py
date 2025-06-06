from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import ast
import re
from time import time

# === Setup ===
load_dotenv()
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Globals ===
model = None
beschikbare_urls = []
client_qdrant = QdrantClient("localhost", port=6333)

def nesting_level(url):
    path = url.split("://", 1)[-1].split("/", 1)[-1]
    return path.count("/")

def hoofdsegment(url):
    parts = url.split("://", 1)[-1].split("/", 1)
    if len(parts) == 2 and "/" in parts[1]:
        return parts[1].split("/")[0]
    elif len(parts) == 2:
        return parts[1]
    else:
        return ""

@app.on_event("startup")
def load_everything():
    global model, beschikbare_urls
    print("📦 Embeddingmodel laden...")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    print("✅ Model geladen.")
    with open("data/url_descriptions.json", encoding="utf-8") as f:
        url_beschrijvingen = json.load(f)
    beschikbare_urls = []
    for u, b in url_beschrijvingen.items():
        lvl = nesting_level(u)
        seg = hoofdsegment(u)
        beschikbare_urls.append({"url": u, "beschrijving": b, "nesting": lvl, "segment": seg})

def extract_text(hit):
    return hit.payload.get("text", "")

def bronvermelding_from_source(hit):
    return hit.payload.get("source", "https://www.gemeentewestland.nl/")

def chat_openai(system_prompt, user_prompt, model="gpt-4o"):
    print(f"\n--- OpenAI call ---\nSYSTEM: {system_prompt}\nUSER: {user_prompt[:200]}...\n")
    response = client_openai.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.2
    )
    print("--- OpenAI raw response ---\n", response)
    return response.choices[0].message.content.strip()

def herformuleer_vraag(vraag: str) -> str:
    prompt = (
        "Je bent een taalmodel dat gebruikersvragen interpreteert alsof ze een zoekopdracht geven "
        "aan een zoekmachine op de website van een Nederlandse gemeente. "
        "Herschrijf de onderstaande vraag zodat deze geschikt is voor semantisch zoeken. "
        "Denk na over wat de gebruiker eigenlijk bedoelt, welke termen hij mogelijk niet noemt maar wel impliciet zijn, "
        "en herschrijf het als een volledige informatieve zoekopdracht in gewone taal.\n\n"
        f"Originele vraag: {vraag}\n\n"
        "Geherformuleerde zoekopdracht:"
    )
    return chat_openai("Je bent een zoekassistent.", prompt)

class VraagInput(BaseModel):
    vraag: str

def apply_nesting_boost(top_hits):
    segment_min = {}
    for score, url, beschrijving, nesting, segment in top_hits:
        if segment not in segment_min or nesting < segment_min[segment]:
            segment_min[segment] = nesting
    NESTING_BONUS = 0.05
    boosted = []
    for score, url, beschrijving, nesting, segment in top_hits:
        rel_nesting = nesting - segment_min.get(segment, 0)
        bonus = -NESTING_BONUS * rel_nesting
        boosted_score = score + bonus
        boosted.append((boosted_score, url, beschrijving))
    boosted.sort(reverse=True)
    return boosted

def parse_llm_json(response):
    try:
        return json.loads(response)
    except Exception:
        try:
            return ast.literal_eval(response)
        except Exception:
            return None

def clean_snippet(snippet):
    # Verwijder Markdown codeblokken/backticks en extraheer innerlijke JSON-string als nodig
    if isinstance(snippet, str):
        # Strip alle markdown-blocks en backticks
        snippet = re.sub(r"^```(?:json)?", "", snippet.strip(), flags=re.MULTILINE)
        snippet = re.sub(r"```$", "", snippet.strip(), flags=re.MULTILINE)
        snippet = snippet.strip()
        # Probeer nog één keer te parsen als het alsnog een JSON-string is
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict) and "snippet" in obj:
                return str(obj["snippet"])
        except Exception:
            pass
    return snippet

@app.post("/chat")
def chat(vraag_input: VraagInput):
    originele_vraag = vraag_input.vraag.strip()
    vraag = herformuleer_vraag(originele_vraag)
    print("Geherformuleerde vraag:", vraag)
    vraag_embedding = model.encode(vraag).tolist()
    t0 = time()

    search_result = client_qdrant.search(
        collection_name="westland-url-descriptions",
        query_vector=vraag_embedding,
        limit=30,
        with_payload=True
    )

    top_hits = []
    for hit in search_result:
        url = hit.payload.get("url")
        beschrijving = hit.payload.get("description", "")
        if url:
            lvl = nesting_level(url)
            seg = hoofdsegment(url)
            top_hits.append((hit.score, url, beschrijving, lvl, seg))
    top_urls = apply_nesting_boost(top_hits)

    lijst_prompt = "\n".join([
        f"{i+1}. {url}: {beschrijving}"
        for i, (_, url, beschrijving) in enumerate(top_urls)
    ])
    selectie_prompt = (
        "Je bent een AI-assistent voor gemeentelijke dienstverlening. "
        "Je krijgt een lijst van pagina's van de gemeente met hun beschrijvingen. "
        "Kies maximaal vijf pagina's waarin het meest waarschijnlijk het antwoord op de vraag te vinden is. "
        "Geef alleen een JSON-lijst met de gekozen URLs (dus niet de beschrijvingen en geen tekst buiten de lijst).\n\n"
        f"Vraag: {vraag}\n\nBeschikbare pagina's:\n{lijst_prompt}"
    )

    try:
        top5_urls_antwoord = chat_openai("Je bent een behulpzame taalassistent.", selectie_prompt)
        print("LLM selectie output:", top5_urls_antwoord)
        gekozen_urls = parse_llm_json(top5_urls_antwoord)
        if not gekozen_urls or not isinstance(gekozen_urls, list):
            raise ValueError("LLM gaf geen geldige lijst")
    except Exception as e:
        print("Fallback: JSON/literal_eval faalde:", e)
        gekozen_urls = [url for _, url, _ in top_urls[:5]]

    print("gekozen_urls:", gekozen_urls)

    alle_chunks = []
    for gekozen_url in gekozen_urls:
        filter = models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=gekozen_url))])
        hits = client_qdrant.scroll("westland-mpnet", scroll_filter=filter, with_payload=True, limit=20)
        hits = hits[0] if isinstance(hits, tuple) else hits
        alle_chunks.extend(hits)

    if not alle_chunks:
        return {
            "resultaat": {
                "raw": "",
                "snippet": "Er is geen specifiek antwoord gevonden op uw vraag in de beschikbare informatie.",
                "extra": [],
                "toelichting": "",
                "bron": ""
            }
        }

    combined_text = "\n\n".join([extract_text(c) for c in alle_chunks])
    bronnen = list({bronvermelding_from_source(c) for c in alle_chunks})

    eindprompt = (
        "Je bent een AI-assistent voor gemeentelijke dienstverlening. "
        "Beantwoord de onderstaande vraag zo feitelijk, duidelijk en bruikbaar mogelijk, "
        "alleen op basis van de gegeven context. Gebruik waar mogelijk letterlijk zinnen uit de context. "
        "Voeg niets toe wat niet in de context staat.\n\n"
        "Splits je antwoord in:\n"
        "- 'snippet': een korte, duidelijke hoofdboodschap\n"
        "- 'extra': eventuele bijzinnen of bijzonderheden (zoals waarschuwingen, uitzonderingen)\n"
        "- 'toelichting': waarom je deze selectie en structuur koos\n\n"
        "Antwoord uitsluitend met de onderstaande JSON (geen uitleg, geen markdown, geen backticks, geen codeblokken, alleen de JSON!):\n"
        "{\n"
        '  "snippet": "...",\n'
        '  "extra": ["...", "..."],\n'
        '  "toelichting": "..." \n'
        "}\n\n"
        f"Context:\n{combined_text}\n\nVraag: {originele_vraag}"
    )

    response = chat_openai("Je bent een behulpzame taalassistent.", eindprompt)
    t1 = time()

    try:
        parsed = json.loads(response)
        snippet = parsed.get("snippet", "")
        snippet = clean_snippet(snippet)
        extra = parsed.get("extra", [])
        toelichting = parsed.get("toelichting", "")
    except Exception as e:
        print("Parse error in eindprompt-antwoord:", e)
        snippet = clean_snippet(response.strip())
        extra = []
        toelichting = ""

    print(f"→ ⏱️ TIMING (s): totaal={t1-t0:.2f}")
    return {
        "resultaat": {
            "raw": response,
            "snippet": snippet,
            "extra": extra,
            "toelichting": toelichting,
            "bron": bronnen[0] if len(bronnen) == 1 else bronnen
        }
    }

@app.post("/debug/top-urls", response_class=PlainTextResponse)
def debug_top_urls(vraag_input: VraagInput):
    originele_vraag = vraag_input.vraag.strip()
    vraag = herformuleer_vraag(originele_vraag)
    vraag_embedding = model.encode(vraag).tolist()

    search_result = client_qdrant.search(
        collection_name="westland-url-descriptions",
        query_vector=vraag_embedding,
        limit=25,
        with_payload=True
    )

    top_hits = []
    for hit in search_result:
        url = hit.payload.get("url")
        beschrijving = hit.payload.get("description", "")
        if url:
            lvl = nesting_level(url)
            seg = hoofdsegment(url)
            top_hits.append((hit.score, url, beschrijving, lvl, seg))
    top_urls = apply_nesting_boost(top_hits)

    lines = [
        f"{i+1}. {url}  (score: {round(score, 4)})"
        for i, (score, url, beschrijving) in enumerate(top_urls)
    ]
    output = [
        f"Herformuleerde zoekopdracht: {vraag}",
        "",
        *lines
    ]
    return "\n".join(output)
