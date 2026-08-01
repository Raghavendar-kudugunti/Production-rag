# FinDocs AI — Production-Minded RAG for Financial Documents

A Retrieval-Augmented Generation system for querying SEC financial filings (10-Ks), built with hybrid search, table-aware ingestion, and a measured evaluation benchmark — not just a demo.

## Why This Project

Industry data shows a large share of RAG systems fail in production, and the dominant causes are consistently **weak retrieval** and **absence of systematic evaluation** — not model quality. This project directly targets both:

- **Hybrid retrieval** (semantic + keyword search) instead of naive vector-only search
- **A measured benchmark**, not just a claim: hybrid retrieval scored **100% (12/12)** accuracy on a test set of real financial questions, versus **91.7% (11/12)** for naive vector-only retrieval on the same questions
- **A head-to-head test against ChatGPT** on the same questions, with no retrieval — see results below

Domain: financial filings, specifically chosen because tables and precise figures (not just prose) are where naive RAG most commonly breaks.

## What It Does

- Answers questions about a company's 10-K filing with citations back to the source section/table
- Extracts both **prose** (Business, Risk Factors, MD&A, etc.) and **tables** (income statement, balance sheet) — tables are converted to Markdown to preserve row/column structure, since naive text extraction jumbles tabular data
- Supports uploading additional documents at runtime through the chat interface
- Grounds every answer strictly in retrieved context, explicitly instructed to say "not found" rather than hallucinate

## Architecture

```
PDF → Extraction (pypdf) → Section Detection (regex, "Item N" structure)
                          → Table Extraction (pdfplumber → Markdown)
    → Chunking (sentence-aware, 256–512 tokens, overlap)
    → Embedding (local sentence-transformers, all-MiniLM-L6-v2)
    → Dual Index: FAISS (vector) + BM25 (keyword)
    → Query time: Hybrid Search (Reciprocal Rank Fusion merge)
    → Generation (Gemini 3.1 Flash Lite, grounded prompt)
    → Answer + cited sources
```

## Evaluation Results

### Retrieval Comparison

12 real questions against Apple's FY2025 10-K, scored by an LLM judge comparing each answer to verified figures from the actual filing.

| Method | Accuracy |
|---|---|
| No retrieval (LLM alone) | See `data/processed/eval_results.json` for full detail |
| Naive (vector search only) | 11/12 (91.7%) |
| **Hybrid (BM25 + vector + RRF merge)** | **12/12 (100%)** |

The one case naive search missed (total operating expenses) is a concrete example of the retrieval failure mode hybrid search is designed to fix: vector search alone surfaced a segment-breakdown table instead of the consolidated income statement; BM25's exact keyword match on "total operating expenses" corrected this in the hybrid result.

**Honest limitation:** this is a small benchmark (12 questions, single document) — a rigorous production benchmark would use 50-100+ questions across multiple filings. This is a proof of concept with real, reproducible measurement, not a claim of exhaustive validation.

### Head-to-Head vs. ChatGPT (No Retrieval)

To test whether retrieval actually matters versus just asking a general-purpose LLM, the same 4 questions were asked directly to ChatGPT (no document provided) and compared against the verified figures from Apple's actual 10-K.

| Question | ChatGPT's Answer | Verified (10-K) | Correct? |
|---|---|---|---|
| Total net sales, FY2025 | $406.7 billion | $416,161 million | ❌ Off by ~$9.5B |
| Net income, FY2025 | $112.0 billion | $112,010 million | ✅ Correct |
| Diluted EPS, FY2025 | $7.55 | $7.46 | ❌ Wrong |
| Services net sales growth, FY2025 | 11.7% (to $107.5B) | 14% (to $109,158M) | ❌ Wrong |

**ChatGPT scored 1/4**, and notably answered all four with equal, unhedged confidence — including the three incorrect ones — citing "Apple's Form 10-K" as its source without apparent verification. This is a direct, reproducible illustration of the failure mode this project is built to address: ungrounded LLMs don't express uncertainty when wrong, they answer confidently regardless. This system's answers, by contrast, come with clickable citations to the exact source section, making every claim independently verifiable.

## Tech Stack

- **Backend:** FastAPI, Python
- **Chat UI:** Chainlit
- **LLM:** Google Gemini (gemini-3.1-flash-lite)
- **Embeddings:** sentence-transformers (local, all-MiniLM-L6-v2) — chosen after hitting persistent free-tier rate limits with hosted embedding APIs; local embeddings eliminate rate-limit risk entirely for the ingestion/upload path
- **Vector store:** FAISS
- **Keyword search:** BM25 (rank_bm25)
- **PDF/table extraction:** pypdf, pdfplumber
- **Orchestration:** LangChain (Google GenAI + community integrations)

## Scope Decisions (What's Not Built, and Why)

Deliberately scoped out to ship a complete, working system rather than a half-finished larger one:

- **Postgres/Neon migration** — currently on FAISS. Neon would add relational metadata filtering and multi-document joins at scale; not needed to prove the core retrieval-quality thesis at this size.
- **Agentic multi-step reasoning** — cut to avoid the reliability risk of an under-tested planning/tool-execution layer; single-hop retrieval + synthesis covers the target use case.
- **Validation nodes (Gatekeeper/Auditor/Strategist)** — the grounded generation prompt provides basic anti-hallucination behavior; a dedicated validation layer is a natural next step.
- **Stress testing / red-teaming** — not yet performed.
- **Hypothetical-questions metadata enrichment** — deferred due to free-tier API rate-limit constraints during development; would improve retrieval further if reintroduced.

## Known Limitations

- Table extraction preserves rows/columns correctly but doesn't always capture column headers (e.g., fiscal year labels) — addressable with layout-aware header detection.
- Document upload uses a structural fallback (paragraph-based chunking) for filings that don't follow the standard "Item N" 10-K format; retrieval quality on non-10-K documents is unverified.
- Document upload is processed in a background thread — very large documents take proportionally longer to become queryable.
- The system's context window usage per query (~2,000 tokens for 5 retrieved chunks) is a tiny fraction of the underlying model's 1M-token window — document size is bounded by ingestion throughput, not context length.

## Running Locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Add a `.env` file with `GEMINI_API_KEY=your_key_here`.

```bash
chainlit run app/chainlit_app.py
```

## Project Structure

```
app/
  ingestion/     # PDF extraction, section detection, chunking, table extraction
  retrieval/     # BM25, local embeddings, hybrid search (RRF)
  generation/    # Gemini client
  evaluation/    # Evaluation harness + test question set
  chainlit_app.py
  main.py        # FastAPI endpoint (alternative to chat UI)
data/
  raw/           # Source PDFs (not committed)
  processed/     # Processed chunks, FAISS index, eval results
```
