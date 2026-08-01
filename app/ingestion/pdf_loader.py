import re
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    full_text = ""
    
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"
        else:
            print(f"Warning: No text extracted from page {page_num + 1}")
    
    return full_text


def detect_item_sections(text: str) -> list[dict]:
    pattern = r"(Item\s+\d+[A-Z]?\.?\s+[A-Z][a-zA-Z\s,&\-'’]+)"
    matches = list(re.finditer(pattern, text))
    
    raw_sections = []
    for i, match in enumerate(matches):
        section_title = match.group().strip()
        start_pos = match.start()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start_pos:end_pos].strip()
        
        raw_sections.append({
            "title": section_title,
            "text": section_text,
            "char_count": len(section_text)
        })
    
    deduped = {}
    for section in raw_sections:
        item_number_match = re.match(r"Item\s+(\d+[A-Z]?)", section["title"])
        if not item_number_match:
            continue
        item_key = item_number_match.group(1)
        
        if item_key not in deduped or section["char_count"] > deduped[item_key]["char_count"]:
            deduped[item_key] = section
    
    return list(deduped.values())

def chunk_by_paragraphs(text: str, max_chars: int = 2000) -> list[dict]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    sections = []
    current_text = ""
    section_num = 1
    
    for para in paragraphs:
        if len(current_text) + len(para) > max_chars and current_text:
            sections.append({
                "title": f"Section {section_num}",
                "text": current_text.strip()
            })
            section_num += 1
            current_text = ""
        current_text += para + "\n\n"
    
    if current_text.strip():
        sections.append({
            "title": f"Section {section_num}",
            "text": current_text.strip()
        })
    
    return sections


if __name__ == "__main__":
    pdf_path = "C:/Users/ragha/production-rag/data/raw/aapl-20250927.pdf"
    print("Extracting text from PDF...")
    raw_text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(raw_text)} characters total.\n")
    
    print("Detecting Item sections...")
    sections = detect_item_sections(raw_text)
    print(f"Found {len(sections)} sections.\n")
    
    for section in sections:
        print(f"{section['title']} — {section['char_count']} characters")