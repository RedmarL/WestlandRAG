# scripts/chunk.py  (Python 3.10+)
import pathlib, json, time, requests
from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer                 # NEW

# ---------- constants ----------
TXT_DIR   = pathlib.Path("data/cleaned_txt")
CHUNK_DIR = pathlib.Path("data/chunks")
CHUNK_DIR.mkdir(exist_ok=True)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer   = AutoTokenizer.from_pretrained(MODEL_NAME)

MAX_TOKENS  = 2048          # keep entire doc if it fits
OVERLAP     = 50            # token overlap when we must split

# ---------- helpers ----------
def url_alive(url: str) -> bool:
    try:
        r = requests.head(url, timeout=5)
        return r.status_code < 400
    except Exception:
        return False

def split_by_tokens(text: str) -> List[str]:
    """Paragraph-aware token splitting."""
    paragraphs = [p for p in text.split("\n") if p.strip()]
    chunks, current, cur_len = [], [], 0

    for para in paragraphs:
        plen = len(tokenizer.encode(para, add_special_tokens=False))
        if cur_len + plen > MAX_TOKENS:
            chunks.append(" ".join(current))
            # start new chunk, keep overlap for continuity
            overlap = current[-OVERLAP:] if OVERLAP < len(current) else current
            current = overlap.copy()
            cur_len = len(tokenizer.encode(" ".join(current), add_special_tokens=False))
        current.append(para)
        cur_len += plen
    if current:
        chunks.append(" ".join(current))
    return chunks

# ---------- main ----------
txt_files = list(TXT_DIR.glob("*.json"))
print(f"🔍 Found {len(txt_files)} cleaned pages.")

for jf in tqdm(txt_files, desc="Chunking"):
    data = json.loads(jf.read_text(encoding="utf-8"))
    text, url = data.get("text", "").strip(), data.get("url") or data.get("source")

    if not text:
        tqdm.write(f"⚠️ Skipped (empty): {jf.name}")
        continue
    if not url_alive(url):
        tqdm.write(f"❌ Unreachable URL: {url}")
        continue

    # Decide whether to split
    tok_len = len(tokenizer.encode(text, add_special_tokens=False))
    if tok_len <= MAX_TOKENS:
        chunks = [text]
    else:
        chunks = split_by_tokens(text)

    # write out each chunk
    for idx, chunk in enumerate(chunks):
        out = CHUNK_DIR / f"{jf.stem}_chunk{idx}.json"
        json.dump({"text": chunk, "source": url},
                  out.open("w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    tqdm.write(f"✅ {jf.name} → {len(chunks)} chunk(s)")
    time.sleep(0.2)

print("\n🎉 All pages processed.")
