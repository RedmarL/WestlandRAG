import pathlib
import requests
import time
import json
from tqdm import tqdm

# 📁 Paden instellen
TXT_DIR = pathlib.Path('data/cleaned_txt')
CHUNK_DIR = pathlib.Path('data/chunks')
CHUNK_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 300

# 🌐 Check of URL werkt
def pagina_beschikbaar(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 404 or "Pagina niet gevonden" in response.text:
            return False
        return True
    except Exception:
        return False

# 🚀 Start verwerkingslus
all_files = list(TXT_DIR.glob('*.json'))
print(f"🔍 Start met chunken van {len(all_files)} bestanden...\n")

for jsonfile in tqdm(all_files, desc="📄 Pagina's verwerken", unit="pagina"):
    with open(jsonfile, encoding='utf-8') as f:
        data = json.load(f)

    text = data.get("text", "")
    url = data.get("url", "")

    if not text or not url:
        tqdm.write(f"⚠️  Overgeslagen (leeg): {jsonfile.name}")
        continue

    if not pagina_beschikbaar(url):
        tqdm.write(f"❌ Niet beschikbaar: {url}")
        continue

    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    for idx, chunk in enumerate(chunks):
        chunk_data = {
            "text": chunk,
            "source": url
        }
        outpath = CHUNK_DIR / f"{jsonfile.stem}_chunk{idx}.json"
        with open(outpath, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2)

    tqdm.write(f"✅ {jsonfile.name} → {len(chunks)} chunks")

    time.sleep(0.2)

print("\n🎉 Alle pagina's zijn gechunked!")
