# (run je eenmalig, opslaan in url_vectors.json)
import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

with open('data/url_descriptions.json', encoding='utf-8') as f:
    url_beschrijvingen = json.load(f)

url_vectors = {}
for url, beschrijving in url_beschrijvingen.items():
    emb = model.encode(beschrijving).tolist()
    url_vectors[url] = emb

with open('data/url_vectors.json', 'w', encoding='utf-8') as f:
    json.dump(url_vectors, f)
