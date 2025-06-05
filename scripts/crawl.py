import requests, pathlib, time, json
import xml.etree.ElementTree as ET
from tqdm import tqdm

RAW_DIR = pathlib.Path('data/raw_pages')
RAW_DIR.mkdir(parents=True, exist_ok=True)

SITEMAP_URL = "https://www.gemeentewestland.nl/sitemap.xml"

# 📥 Haal sitemap op
sitemap_xml = requests.get(SITEMAP_URL, timeout=20).text
root = ET.fromstring(sitemap_xml)

# 🔗 Verzamel alle URLs uit sitemap
urls = [loc.text for loc in root.iter('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')]
print(f"Totaal gevonden: {len(urls)} pagina's.")

for url in tqdm(urls):
    slug = url.replace("https://www.gemeentewestland.nl/", "").strip("/").replace("/", "_")
    path = RAW_DIR / f"{slug}.json"

    if path.exists():
        continue  # skip als bestand al bestaat

    try:
        html = requests.get(url, timeout=10).text
        record = {
            "url": url,
            "html": html
        }
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(0.5)  # wees vriendelijk voor de server
    except Exception as e:
        print(f"Fout bij {url}: {e}")
