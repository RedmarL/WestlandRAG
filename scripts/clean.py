import trafilatura
import json
import pathlib

RAW_DIR = pathlib.Path('data/raw_pages')
TXT_DIR = pathlib.Path('data/cleaned_txt')
TXT_DIR.mkdir(exist_ok=True)

for jsonfile in RAW_DIR.glob('*.json'):
    with open(jsonfile, encoding='utf-8') as f:
        data = json.load(f)

    html = data.get('html', '')
    url = data.get('url', '')

    if not html:
        continue

    text = trafilatura.extract(html)
    if text:
        outpath = TXT_DIR / jsonfile.with_suffix('.txt').name
        with open(outpath, 'w', encoding='utf-8') as f:
            f.write(text)
