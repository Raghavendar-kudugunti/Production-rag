import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from google import genai

from retrieval.bm25_search import load_chunks, build_bm25_index
from retrieval.hybrid_search import hybrid_search

load_dotenv()

app = FastAPI()

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

chunks = load_chunks("data/processed/apple_2025_chunks.json")
bm25 = build_bm25_index(chunks)
vector_store = FAISS.load_local(
    "data/processed/faiss_index",
    embeddings_model,
    allow_dangerous_deserialization=True
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    retrieved_chunks = hybrid_search(request.question, vector_store, bm25, chunks, top_k=5)
    
    context = "\n\n".join(
        f"[Source: {c['section_title']}]\n{c['text']}" for c in retrieved_chunks
    )
    
    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say so clearly.

Context:
{context}

Question: {request.question}

Answer:"""
    
    response = gemini_client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )
    
    sources = [c["section_title"] for c in retrieved_chunks]
    return AskResponse(answer=response.text, sources=sources)


@app.get("/")
def root():
    return {"status": "running", "chunks_loaded": len(chunks)}