import pathlib
import json
import uuid
import re
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from transformers import AutoTokenizer
from tqdm import tqdm
from datetime import datetime, timezone

# ---------- Constants ----------
RAW_PAGES_DIR = pathlib.Path("data/raw_pages")
CHUNK_DIR = pathlib.Path("data/chunks")
CHUNK_DIR.mkdir(exist_ok=True)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

MAX_TOKENS = 512
OVERLAP = 100

# ---------- Content Extraction ----------
def extract_content_from_html(html_content: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html_content, 'html.parser')

    document_title = soup.title.string.strip() if soup.title and soup.title.string else "Geen titel"
    extracted_date = None
    author_department = "Gemeente Westland"

    # Try meta tags for date
    date_meta = soup.find('meta', attrs={"property": "article:modified_time"}) or \
                soup.find('meta', attrs={"property": "article:published_time"})
    if date_meta and date_meta.get("content"):
        extracted_date = date_meta["content"]

    content_blocks = []

    # Grab main title (fallback if <title> is vague)
    h1 = soup.find("h1")
    if h1:
        content_blocks.append(h1.get_text(strip=True))

    # Gemeente-specific structure
    for section in soup.select("div.ce-bodytext, div.collapsible__content"):
        text = section.get_text(separator="\n", strip=True)
        if text and len(text) > 20:
            content_blocks.append(text)

    # Also add <h2> and <h3> headings for structure
    for heading in soup.find_all(["h2", "h3"]):
        heading_text = heading.get_text(strip=True)
        if heading_text and heading_text not in content_blocks:
            content_blocks.append(heading_text)

    # Clean and deduplicate
    cleaned_blocks = []
    seen = set()
    for block in content_blocks:
        block = re.sub(r'\n+', '\n', block).strip()
        if block and block not in seen:
            seen.add(block)
            cleaned_blocks.append(block)

    return {
        "content_blocks": cleaned_blocks,
        "document_title": document_title,
        "extracted_date": extracted_date,
        "author_department": author_department
    }

# ---------- Chunking ----------
def split_content_blocks_by_tokens(doc_data: Dict[str, Any], max_tokens: int, overlap: int, doc_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks = []
    current_chunk_text = ""
    current_chunk_tokens = 0

    title_text = doc_data.get("document_title", "")
    initial_prefix = f"Titel: {title_text}\n\n" if title_text else ""
    is_first = True

    for block_text in doc_data["content_blocks"]:
        text = initial_prefix + block_text if is_first and not current_chunk_text else block_text
        tokens = len(tokenizer.encode(text))

        if current_chunk_tokens + tokens > max_tokens:
            if current_chunk_text:
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": current_chunk_text.strip(),
                    "token_count": current_chunk_tokens,
                    **doc_metadata
                })
                is_first = False
                overlap_text = tokenizer.decode(tokenizer.encode(current_chunk_text)[-overlap:])
                current_chunk_text = overlap_text + "\n" + block_text
                current_chunk_tokens = len(tokenizer.encode(current_chunk_text))
            else:
                words = block_text.split()
                part = []
                prefix = initial_prefix if is_first else ""
                for word in words:
                    part.append(word)
                    part_text = prefix + " ".join(part)
                    if len(tokenizer.encode(part_text)) > max_tokens and len(part) > 1:
                        chunks.append({
                            "id": str(uuid.uuid4()),
                            "text": prefix + " ".join(part[:-1]),
                            "token_count": len(tokenizer.encode(prefix + " ".join(part[:-1]))),
                            **doc_metadata
                        })
                        part = [word]
                        prefix = ""
                        is_first = False
                if part:
                    chunks.append({
                        "id": str(uuid.uuid4()),
                        "text": prefix + " ".join(part),
                        "token_count": len(tokenizer.encode(prefix + " ".join(part))),
                        **doc_metadata
                    })
                    is_first = False
                current_chunk_text = ""
                current_chunk_tokens = 0
        else:
            if current_chunk_text:
                current_chunk_text += "\n" + block_text
            else:
                current_chunk_text = text
            current_chunk_tokens = len(tokenizer.encode(current_chunk_text))

    if current_chunk_text:
        chunks.append({
            "id": str(uuid.uuid4()),
            "text": current_chunk_text.strip(),
            "token_count": current_chunk_tokens,
            **doc_metadata
        })

    return chunks

# ---------- Main ----------
def main():
    print(f"🔍 Starting chunking process in {RAW_PAGES_DIR}...")
    files = list(RAW_PAGES_DIR.glob("*.json"))
    if not files:
        print("⚠️ No files found. Run crawl.py first.")
        return

    for file_path in tqdm(files, desc="Chunking files"):
        try:
            with open(file_path, encoding="utf-8") as f:
                record = json.load(f)

            source_url = record.get("url")
            html_content = record.get("html") or record.get("content")
            last_modified = record.get("last_modified_header")
            inferred_category = record.get("category", "Algemeen")

            if not source_url or not html_content:
                tqdm.write(f"⚠️ Skipped (missing URL or HTML): {file_path.name}")
                continue

            parsed = extract_content_from_html(html_content)
            if not parsed["content_blocks"]:
                tqdm.write(f"⚠️ Skipped (no content found): {file_path.name}")
                continue

            doc_metadata = {
                "source_url": source_url,
                "document_type": "html_page",
                "document_title": parsed["document_title"],
                "document_date": parsed["extracted_date"] or last_modified,
                "category": inferred_category,
                "author_department": parsed["author_department"]
            }

            chunks = split_content_blocks_by_tokens(
                {"document_title": parsed["document_title"], "content_blocks": parsed["content_blocks"]},
                MAX_TOKENS,
                OVERLAP,
                doc_metadata
            )

            for i, chunk in enumerate(chunks):
                out_path = CHUNK_DIR / f"{file_path.stem}_chunk{i}.json"
                with open(out_path, "w", encoding="utf-8") as out_f:
                    json.dump(chunk, out_f, ensure_ascii=False, indent=2)

            tqdm.write(f"✅ {file_path.name} → {len(chunks)} chunks")

        except Exception as e:
            tqdm.write(f"❌ Error processing {file_path.name}: {e}")

    print("✅ Chunking complete.")

if __name__ == "__main__":
    main()
