import pathlib
import json
import time
import requests
from typing import List, Dict, Union, Any, Optional
from tqdm import tqdm
from transformers import AutoTokenizer
from bs4 import BeautifulSoup, Tag
import uuid
from datetime import datetime, timezone # Use timezone for consistency
import re # For regex if needed for date/author extraction

# ---------- constants ----------
RAW_PAGES_DIR = pathlib.Path("data/raw_pages") # Directory containing HTML JSON files
CHUNK_DIR = pathlib.Path("data/chunks")
CHUNK_DIR.mkdir(exist_ok=True)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

MAX_TOKENS = 512
OVERLAP = 100

# ---------- helpers ----------

def extract_content_from_html(html_content: str) -> Dict[str, Any]:
    """
    Parses HTML to extract structured content blocks (headings, paragraphs, lists, tables)
    along with the main document title, publication/last-modified date, and author/department.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    document_title = soup.title.string if soup.title else "Geen titel"
    extracted_date: Optional[str] = None
    author_department: Optional[str] = None

    # --- Date Extraction ---
    # Try to get from common meta tags
    date_meta_tags = [
        {'property': 'article:published_time'},
        {'property': 'article:modified_time'},
        {'name': 'date'},
        {'name': 'pubdate'},
        {'itemprop': 'datePublished'},
        {'itemprop': 'dateModified'}
    ]
    for attrs in date_meta_tags:
        meta_tag = soup.find('meta', attrs=attrs)
        if meta_tag and 'content' in meta_tag.attrs:
            extracted_date = meta_tag['content']
            break
    
    # Fallback: look for a specific span/div often used for publication dates
    if not extracted_date:
        date_span = soup.find('span', class_=re.compile(r'date|published|modified', re.IGNORECASE))
        if date_span:
            # Attempt to parse text in the span, e.g., 'Gepubliceerd op 10 juni 2024'
            date_text = date_span.get_text(strip=True)
            # More robust date parsing might be needed here, e.g., dateutil.parser.parse
            # For now, we'll just take the text, assuming it's somewhat parsable later.
            if re.search(r'\d{1,2}\s+(jan|feb|mar|apr|mei|jun|jul|aug|sep|okt|nov|dec)\w*\s+\d{4}', date_text, re.IGNORECASE):
                extracted_date = date_text.strip()


    # --- Author/Department Extraction ---
    # This part is highly dependent on the website's specific HTML structure.
    # You'll need to inspect pages on gemeentewestland.nl to find reliable selectors.
    # Examples:
    # Look for specific divs or spans known to contain author/department info
    department_tag = soup.find('span', class_='department-name') # Example class
    if department_tag:
        author_department = department_tag.get_text(strip=True)
    
    if not author_department:
        # Check common footer or header areas that might mention departments
        footer_text = soup.find('footer')
        if footer_text:
            match = re.search(r"(Afdeling|Dienst|Team):\s*([A-Za-z\s]+)", footer_text.get_text(), re.IGNORECASE)
            if match:
                author_department = match.group(2).strip()
    
    # Fallback: if no specific department found, maybe default to "Gemeente Westland" or "Onbekend"
    if not author_department:
        author_department = "Gemeente Westland" # Defaulting if not found

    content_blocks = []
    # A simple approach: extract text from common content tags
    # You might want to refine this based on Westland's website structure to avoid boilerplate.
    # Consider filtering out navigation, footers, headers, sidebars.
    # Example: filter by parent element IDs/classes or based on tag hierarchy.
    main_content_area = soup.find('main') or soup.find('div', class_='main-content') # Find main content div
    if main_content_area:
        for tag in main_content_area.find_all(['h1', 'h2', 'h3', 'p', 'li', 'table', 'div'], recursive=False):
            # Further refine filtering if needed, e.g., skip empty tags or specific classes
            if tag.name == 'div' and not tag.get_text(strip=True):
                continue # Skip empty divs
            content_blocks.append(tag.get_text(separator="\n", strip=True))
    else: # Fallback if no main content area found
        for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'li', 'table'], class_=lambda x: 'nav' not in str(x).lower() and 'footer' not in str(x).lower()):
            content_blocks.append(tag.get_text(separator="\n", strip=True))

    # Basic cleanup: remove excessive newlines
    content_blocks = [re.sub(r'\n\s*\n', '\n', block).strip() for block in content_blocks if block.strip()]


    return {
        "content_blocks": content_blocks,
        "document_title": document_title,
        "extracted_date": extracted_date,
        "author_department": author_department
    }

def split_content_blocks_by_tokens(
    doc_data: Dict[str, Any], # Now expects dict with "document_title" and "content_blocks"
    max_tokens: int,
    overlap: int,
    doc_metadata: Dict[str, Any] # Contains document-level metadata like source_url, category etc.
) -> List[Dict[str, Any]]:
    """
    Splits content blocks into chunks based on token limits,
    ensuring each chunk retains full document metadata.
    """
    chunks = []
    current_chunk_text = ""
    current_chunk_tokens = 0
    
    # Prepend title to every chunk
    title_text = doc_data.get("document_title", "")
    if title_text:
        title_text = f"Titel: {title_text}\n\n"
    
    for block_text in doc_data["content_blocks"]:
        block_tokens = len(tokenizer.encode(block_text))

        # Check if adding the current block exceeds MAX_TOKENS
        # Account for title tokens if prepending
        if current_chunk_tokens + block_tokens > max_tokens:
            # If current chunk has content, finalize it before adding new block
            if current_chunk_text:
                chunk_id = str(uuid.uuid4())
                chunks.append({
                    "id": chunk_id,
                    "text": current_chunk_text.strip(),
                    "token_count": current_chunk_tokens,
                    **doc_metadata # Spread document-level metadata into each chunk
                })

                # Prepare for next chunk with overlap
                overlap_text = tokenizer.decode(tokenizer.encode(current_chunk_text)[-overlap:])
                current_chunk_text = overlap_text + "\n" + block_text
                current_chunk_tokens = len(tokenizer.encode(current_chunk_text))
            else:
                # If a single block is larger than MAX_TOKENS, split it
                # This is a simplified split; for very large blocks, a more
                # sophisticated recursive splitting might be needed.
                sub_blocks = []
                words = block_text.split()
                temp_block = []
                for word in words:
                    temp_block.append(word)
                    if len(tokenizer.encode(" ".join(temp_block))) > max_tokens:
                        sub_blocks.append(" ".join(temp_block[:-1]))
                        temp_block = [word]
                if temp_block:
                    sub_blocks.append(" ".join(temp_block))
                
                for sub_block in sub_blocks:
                    chunk_id = str(uuid.uuid4())
                    chunks.append({
                        "id": chunk_id,
                        "text": title_text + sub_block.strip(), # Prepend title to sub-blocks too
                        "token_count": len(tokenizer.encode(title_text + sub_block)),
                        **doc_metadata
                    })
                current_chunk_text = "" # Reset after splitting large block
                current_chunk_tokens = 0

        else:
            current_chunk_text += (f"\n{block_text}" if current_chunk_text else block_text)
            current_chunk_tokens += block_tokens

    # Add the last accumulated chunk if it has content
    if current_chunk_text:
        chunk_id = str(uuid.uuid4())
        chunks.append({
            "id": chunk_id,
            "text": current_chunk_text.strip(),
            "token_count": current_chunk_tokens,
            **doc_metadata
        })
    
    # Prepend title to all chunks
    final_chunks = []
    for chunk in chunks:
        chunk['text'] = title_text + chunk['text']
        chunk['token_count'] = len(tokenizer.encode(chunk['text']))
        final_chunks.append(chunk)

    return final_chunks


# ---------- Main Processing Logic ----------

def main():
    print(f"Starting chunking process for files in {RAW_PAGES_DIR}...")

    raw_html_files = list(RAW_PAGES_DIR.glob('*.json'))
    if not raw_html_files:
        print(f"No raw HTML files found in {RAW_PAGES_DIR}. Please run crawl.py first.")
        return

    for file_path in tqdm(raw_html_files, desc="Processing HTML files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                record = json.load(f)

            source_url = record.get("url")
            html_content = record.get("html")
            
            # Retrieve new metadata added by crawl.py
            last_modified_header = record.get("last_modified_header")
            inferred_category = record.get("category", "Algemeen") # Default to Algemeen if not found

            if not source_url or not html_content:
                tqdm.write(f"⚠️ Skipped (missing URL or HTML content): {file_path.name}")
                continue

            # NEW: Extract content blocks, document title, date, and author/department from HTML
            parsed_data = extract_content_from_html(html_content)
            content_blocks = parsed_data["content_blocks"]
            document_title = parsed_data["document_title"]
            extracted_date = parsed_data["extracted_date"]
            author_department = parsed_data["author_department"]

            if not content_blocks:
                tqdm.write(f"⚠️ Skipped (no content blocks extracted): {file_path.name}")
                continue
            
            # Determine the best document date to use
            final_document_date = extracted_date # Prioritize date found in meta tags/content
            if not final_document_date and last_modified_header:
                final_document_date = last_modified_header # Fallback to HTTP header date
            
            # Prepare document-level metadata to pass to chunking function
            doc_metadata = {
                "source_url": source_url,
                "document_type": "html_page", # Explicitly set for HTML documents
                "document_title": document_title, # Pass the extracted title
                "document_date": final_document_date, # Use the best available date
                "category": inferred_category, # From crawl.py
                "author_department": author_department, # Extracted from HTML
            }

            # Split content blocks into final chunks, passing all document metadata
            chunks = split_content_blocks_by_tokens(
                {"document_title": document_title, "content_blocks": content_blocks},
                MAX_TOKENS,
                OVERLAP,
                doc_metadata # Pass the rich metadata
            )

            # Write out each chunk as a separate JSON file
            for idx, chunk in enumerate(chunks):
                out_path = CHUNK_DIR / f"{file_path.stem}_chunk{idx}.json"
                # Ensure the entire chunk dictionary, including all metadata, is dumped
                json.dump(chunk, out_path.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
            tqdm.write(f"✅ Processed {file_path.name} → {len(chunks)} chunks.")

        except json.JSONDecodeError:
            tqdm.write(f"❌ Failed to parse JSON from {file_path.name}. Check if it's valid JSON.")
        except Exception as e:
            tqdm.write(f"❌ Error processing {file_path.name}: {e}")

    print("Chunking process complete.")

if __name__ == "__main__":
    main()