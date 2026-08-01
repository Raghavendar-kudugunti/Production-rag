import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import chainlit as cl
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from google import genai

from ingestion.pdf_loader import extract_text_from_pdf, detect_item_sections, chunk_by_paragraphs
from ingestion.chunker import chunk_section, count_tokens
from ingestion.table_extractor import extract_tables_from_pdf
from retrieval.bm25_search import load_chunks, build_bm25_index
from retrieval.embedder import embeddings_model
from retrieval.hybrid_search import hybrid_search

load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

CHUNKS_PATH = "data/processed/apple_2025_chunks.json"
INDEX_PATH = "data/processed/faiss_index"

chunks = load_chunks(CHUNKS_PATH)
bm25 = build_bm25_index(chunks)
vector_store = FAISS.load_local(INDEX_PATH, embeddings_model, allow_dangerous_deserialization=True)


def process_uploaded_pdf(file_path: str, doc_id: str):
    raw_text = extract_text_from_pdf(file_path)
    sections = detect_item_sections(raw_text)

    used_fallback = False
    if not sections:
        sections = chunk_by_paragraphs(raw_text)
        used_fallback = True

    new_chunks = []
    counter = 0

    for section in sections:
        for chunk_text in chunk_section(section["text"]):
            counter += 1
            new_chunks.append({
                "chunk_id": f"{doc_id}_{counter}",
                "company": doc_id,
                "filing_year": "N/A",
                "section_title": section["title"],
                "text": chunk_text,
                "chunk_type": "prose",
                "token_count": count_tokens(chunk_text),
                "hypothetical_questions": []
            })

    tables = extract_tables_from_pdf(file_path)
    for table in tables:
        counter += 1
        new_chunks.append({
            "chunk_id": f"{doc_id}_{counter}",
            "company": doc_id,
            "filing_year": "N/A",
            "section_title": f"Table (page {table['page']})",
            "text": table["markdown"],
            "chunk_type": "table",
            "token_count": count_tokens(table["markdown"]),
            "hypothetical_questions": []
        })

    return new_chunks, used_fallback


def embed_and_index(new_chunks: list[dict]):
    texts = [c["text"] for c in new_chunks]
    vectors = embeddings_model.embed_documents(texts)

    text_embedding_pairs = list(zip(texts, vectors))
    metadatas = [
        {
            "chunk_id": c["chunk_id"],
            "company": c["company"],
            "filing_year": c["filing_year"],
            "section_title": c["section_title"],
            "token_count": c["token_count"]
        }
        for c in new_chunks
    ]
    return text_embedding_pairs, metadatas


async def add_document_to_index(file_path: str, doc_id: str):
    global chunks, bm25, vector_store

    status = cl.Message(content=f"📄 Reading **{doc_id}**...")
    await status.send()

    new_chunks, used_fallback = await cl.make_async(process_uploaded_pdf)(file_path, doc_id)

    status.content = "🔍 Understanding the document structure..."
    await status.update()

    text_embedding_pairs, metadatas = await cl.make_async(embed_and_index)(new_chunks)

    vector_store.add_embeddings(text_embeddings=text_embedding_pairs, metadatas=metadatas)
    chunks.extend(new_chunks)
    bm25 = build_bm25_index(chunks)

    vector_store.save_local(INDEX_PATH)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    fallback_note = "\n\n*(Note: this document has a different structure than a standard 10-K, so I used a general reading approach — results may be slightly less precise.)*" if used_fallback else ""

    status.content = (
        f"✅ **{doc_id}** is ready! You can now ask questions about it — "
        f"try something like *\"What was total revenue?\"* or *\"Summarize the key risk factors.\"*"
        f"{fallback_note}"
    )
    await status.update()

@cl.on_chat_start
async def start():
    await cl.Message(
        content="👋 Welcome! I have Apple's FY2025 10-K loaded and ready. Ask me anything about it, or upload your own PDF (📎) to add another document."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    pdf_elements = [e for e in message.elements if e.mime == "application/pdf"]

    if pdf_elements:
        for element in pdf_elements:
            doc_id = os.path.splitext(element.name)[0]
            await add_document_to_index(element.path, doc_id)

        if not message.content.strip():
            return

    question = message.content
    if not question.strip():
        return

    retrieved_chunks = hybrid_search(question, vector_store, bm25, chunks, top_k=5)

    context = "\n\n".join(
        f"[Source: {c['section_title']}]\n{c['text']}" for c in retrieved_chunks
    )

    prompt = f"""You are a financial analyst assistant. Answer the question using ONLY the context below, but synthesize a clear, well-explained answer in your own words rather than copying fragments.

Include specific figures/facts, with brief context (e.g., what drove a change) if supported by the source. If the answer isn't in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

    response = gemini_client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )

    source_elements = [
        cl.Text(
            name=f"Source {i+1}",
            content=f"**{c['section_title'].splitlines()[0]}**\n\n{c['text'][:300]}...",
            display="side"
        )
        for i, c in enumerate(retrieved_chunks)
    ]

    await cl.Message(content=response.text, elements=source_elements).send()