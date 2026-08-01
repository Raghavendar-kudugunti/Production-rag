import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.pdf_loader import extract_text_from_pdf, detect_item_sections
from ingestion.chunker import chunk_section, count_tokens
from ingestion.table_extractor import extract_tables_from_pdf


def process_document(pdf_path: str, company: str, filing_year: str, save_path: str) -> list[dict]:
    print(f"Extracting text from {pdf_path}...")
    raw_text = extract_text_from_pdf(pdf_path)
    
    print("Detecting sections...")
    sections = detect_item_sections(raw_text)
    print(f"Found {len(sections)} sections.\n")
    
    all_chunks = []
    chunk_id_counter = 0
    
    for section in sections:
        section_chunks = chunk_section(section["text"])
        print(f"{section['title']}: {len(section_chunks)} chunks")
        
        for chunk_text in section_chunks:
            chunk_id_counter += 1
            all_chunks.append({
                "chunk_id": f"{company}_{filing_year}_{chunk_id_counter}",
                "company": company,
                "filing_year": filing_year,
                "section_title": section["title"],
                "text": chunk_text,
                "chunk_type": "prose",
                "token_count": count_tokens(chunk_text),
                "hypothetical_questions": []
            })
    
    print("\nExtracting tables...")
    tables = extract_tables_from_pdf(pdf_path)
    print(f"Found {len(tables)} tables.\n")
    
    for table in tables:
        chunk_id_counter += 1
        all_chunks.append({
            "chunk_id": f"{company}_{filing_year}_{chunk_id_counter}",
            "company": company,
            "filing_year": filing_year,
            "section_title": f"Table (page {table['page']})",
            "text": table["markdown"],
            "chunk_type": "table",
            "token_count": count_tokens(table["markdown"]),
            "hypothetical_questions": []
        })
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    return all_chunks


if __name__ == "__main__":
    output_path = "data/processed/apple_2025_chunks.json"
    
    chunks = process_document(
        pdf_path="data/raw/aapl-20250927.pdf",
        company="Apple",
        filing_year="2025",
        save_path=output_path
    )
    
    prose_count = sum(1 for c in chunks if c["chunk_type"] == "prose")
    table_count = sum(1 for c in chunks if c["chunk_type"] == "table")
    print(f"\nDone. Saved {len(chunks)} total chunks ({prose_count} prose, {table_count} table) to {output_path}")