import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

st_model = SentenceTransformer("all-MiniLM-L6-v2")


class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return st_model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return st_model.encode([text], show_progress_bar=False)[0].tolist()


embeddings_model = LocalEmbeddings()


def load_chunks(chunks_path: str) -> list[dict]:
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_vector_store(chunks: list[dict]) -> FAISS:
    texts = [c["text"] for c in chunks]
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
        for c in chunks
    ]

    vector_store = FAISS.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=embeddings_model,
        metadatas=metadatas
    )
    return vector_store


if __name__ == "__main__":
    chunks = load_chunks("data/processed/apple_2025_chunks.json")

    print(f"Embedding all {len(chunks)} chunks locally (no API calls)...")
    vector_store = build_vector_store(chunks)
    print("Vector store built successfully.")

    vector_store.save_local("data/processed/faiss_index")
    print("Saved FAISS index to data/processed/faiss_index\n")

    query = "What were Apple's total net sales?"
    results = vector_store.similarity_search(query, k=3)

    print(f"Query: {query}\n")