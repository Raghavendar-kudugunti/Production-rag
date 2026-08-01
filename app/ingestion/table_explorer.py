import pdfplumber

pdf_path = "data/raw/aapl-20250927.pdf"

with pdfplumber.open(pdf_path) as pdf:
    for page_num, page in enumerate(pdf.pages):
        tables = page.extract_tables()
        for t_idx, table in enumerate(tables):
            table_text = " ".join(
                " ".join(cell or "" for cell in row) for row in table
            )
            if "net sales" in table_text.lower():
                print(f"=== Page {page_num + 1}, Table {t_idx + 1} ===")
                for row in table:
                    print(row)
                print()