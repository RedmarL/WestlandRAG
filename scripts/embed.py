from sentence_transformers import SentenceTransformer
import pathlib, pickle, json

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
CHUNK_DIR = pathlib.Path('data/chunks')
EMB_DIR = pathlib.Path('data/embeddings')
EMB_DIR.mkdir(exist_ok=True)

for chunkfile in CHUNK_DIR.glob('*.json'):
    with open(chunkfile, encoding='utf-8') as f:
        data = json.load(f)

    text = data.get('text', '')
    source = data.get('source', 'https://www.gemeentewestland.nl/')
    if not text:
        continue

    vec = model.encode(text)

    with open(EMB_DIR / chunkfile.with_suffix('.pkl').name, 'wb') as f:
        pickle.dump({
            'text': text,
            'source': source,
            'embedding': vec
        }, f)
