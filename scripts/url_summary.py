import json
from collections import defaultdict
import ollama

# === 1. Laad alle chunks ===
with open('data/chunks/all_chunks.json', encoding='utf-8') as f:
    chunks = json.load(f)

# === 2. Groepeer chunks per URL ===
urls = defaultdict(list)
for c in chunks:
    urls[c['source']].append(c['text'])

# === 3. Maak beschrijvingen via LLM ===
beschrijvingen = {}

for url, texts in urls.items():
    # Combineer eerste 2 chunks, of minder als er minder zijn
    context = "\n".join(texts[:2])
    prompt = (
        "Vat de onderstaande tekst samen in maximaal één duidelijke zin. "
        "Schrijf het zo dat het geschikt is als korte omschrijving van een pagina in een gemeentelijke kennisbank. "
        "Noem alleen het hoofdonderwerp, niet de details. "
        "Hier is de tekst:\n\n"
        f"{context}"
    )

    response = ollama.chat(
        model='mistral',
        messages=[{"role": "user", "content": prompt}]
    )
    beschrijving = response['message']['content'].strip().replace('\n', ' ')
    beschrijvingen[url] = beschrijving
    print(f"{url}: {beschrijving}")

# === 4. Sla de resultaten op ===
with open('url_descriptions.json', 'w', encoding='utf-8') as f:
    json.dump(beschrijvingen, f, ensure_ascii=False, indent=2)

print("Klaar! Alle url-beschrijvingen opgeslagen in url_descriptions.json.")
