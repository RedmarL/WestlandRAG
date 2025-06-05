import json
import glob
import os

all_chunks = []

# Vind alle .json bestanden in data/chunks
for filepath in glob.glob('data/chunks/*.json'):
    with open(filepath, encoding='utf-8') as f:
        data = json.load(f)
        # Elk bestand kan een dict of een lijst zijn
        if isinstance(data, dict):
            all_chunks.append(data)
        elif isinstance(data, list):
            all_chunks.extend(data)
        else:
            print(f"Onbekend formaat in {filepath}")

print(f"Totaal aantal chunks: {len(all_chunks)}")

with open("all_chunks.json", "w", encoding="utf-8") as out:
    json.dump(all_chunks, out, ensure_ascii=False, indent=2)
