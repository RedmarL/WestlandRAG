from sentence_transformers import SentenceTransformer 
import pathlib, pickle, json
from tqdm import tqdm

# 🧠 Model laden
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 📁 Mappen instellen
CHUNK_DIR = pathlib.Path('data/chunks')
EMB_DIR = pathlib.Path('data/embeddings')
EMB_DIR.mkdir(exist_ok=True)

# 📦 Start embedding
chunkfiles = list(CHUNK_DIR.glob('*.json'))
print(f"🔢 Start embedding van {len(chunkfiles)} chunks...\n")

for chunkfile in tqdm(chunkfiles, desc="🔗 Chunks embedden", unit="chunk"):
    with open(chunkfile, encoding='utf-8') as f:
        data = json.load(f)

    text = data.get('text', '').strip()
    source = data.get('source', 'https://www.gemeentewestland.nl/')
    
    if not text or len(text) < 10:
        tqdm.write(f"⚠️  Overgeslagen: {chunkfile.name} (tekst ontbreekt of is te kort)")
        continue

    vec = model.encode(text)

    outpath = EMB_DIR / f"{chunkfile.stem}.pkl"
    with open(outpath, 'wb') as f:
        pickle.dump({
            'text': text,
            'source': source,
            'embedding': vec
        }, f)

print("\n✅ Klaar! Alle geldige chunks zijn succesvol geëmbed.")
