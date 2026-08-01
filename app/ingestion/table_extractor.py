import pdfplumber


def clean_row(row: list) -> list[str]:
    return [(cell or "").strip() for cell in row]


def table_to_markdown(table: list[list]) -> str:
    cleaned_rows = [clean_row(row) for row in table]
    cleaned_rows = [row for row in cleaned_rows if any(cell for cell in row)]
    
    if not cleaned_rows:
        return ""
    
    num_cols = max(len(row) for row in cleaned_rows)
    
    markdown_lines = []
    for i, row in enumerate(cleaned_rows):
        padded_row = row + [""] * (num_cols - len(row))
        markdown_lines.append("| " + " | ".join(padded_row) + " |")
        
        if i == 0:
            markdown_lines.append("|" + "---|" * num_cols)
    
    return "\n".join(markdown_lines)


def extract_tables_from_pdf(pdf_path: str) -> list[dict]:
    extracted_tables = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for t_idx, table in enumerate(tables):
                markdown = table_to_markdown(table)
                if markdown and len(markdown) > 50:
                    extracted_tables.append({
                        "page": page_num + 1,
                        "table_index": t_idx,
                        "markdown": markdown,
                        "row_count": len(table)
                    })
    
    return extracted_tables


if __name__ == "__main__":
    tables = extract_tables_from_pdf("data/raw/aapl-20250927.pdf")
    print(f"Extracted {len(tables)} tables total.\n")
    
    income_statement = next(t for t in tables if t["page"] == 33)
    print("=== Page 33 Table (Income Statement) as Markdown ===\n")
    print(income_statement["markdown"])