import pathlib
import requests
import time
from tqdm import tqdm

TXT_DIR = pathlib.Path('data/cleaned_txt')
CHUNK_DIR = pathlib.Path('data/chunks'); CHUNK_DIR.mkdir(exist_ok=True)
CHUNK_SIZE = 300

def get_url_from_filename(filename):
    # Verwijder .txt en vervang underscores met slashes
    clean_path = filename.stem.replace("_", "/")
    return f"https://www.gemeentewestland.nl/{clean_path}"

def pagina_beschikbaar(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 404:
            return False
        if "Pagina niet gevonden" in response.text:
            return False
        return True
    except Exception:
        return False

for txtfile in tqdm(list(TXT_DIR.glob('*.txt')), desc="Chunks maken"):
    url = get_url_from_filename(txtfile)

    if not pagina_beschikbaar(url):
        print(f"⚠️  Overgeslagen (niet beschikbaar): {url}")
        continue

    text = txtfile.read_text('utf-8')
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    for idx, chunk in enumerate(chunks):
        outfile = CHUNK_DIR / f"{txtfile.stem}_chunk{idx}.txt"
        outfile.write_text(chunk, 'utf-8')

    time.sleep(0.2)  # om server te ontzien
