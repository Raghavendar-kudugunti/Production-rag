import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.vectorstores import FAISS

from retrieval.bm25_search import load_chunks, build_bm25_index, bm25_search
from retrieval.embedder import embeddings_model


def vector_search(vector_store, query: str, k: int = 10):
    results = vector_store.similarity_search(query, k=k)
    return [r.metadata["chunk_id"] for r in results]


def reciprocal_rank_fusion(vector_ids: list[str], bm25_ids: list[str], k: int = 60) -> list[str]:
    scores = {}

    for rank, chunk_id in enumerate(vector_ids):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)

    for rank, chunk_id in enumerate(bm25_ids):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, score in ranked]


def hybrid_search(query: str, vector_store, bm25, chunks: list[dict], top_k: int = 5) -> list[dict]:
    vector_ids = vector_search(vector_store, query, k=10)

    bm25_results = bm25_search(bm25, chunks, query, k=10)
    bm25_ids = [r["chunk"]["chunk_id"] for r in bm25_results]

    fused_ids = reciprocal_rank_fusion(vector_ids, bm25_ids)

    chunk_lookup = {c["chunk_id"]: c for c in chunks}
    return [chunk_lookup[cid] for cid in fused_ids[:top_k] if cid in chunk_lookup]


if __name__ == "__main__":
    chunks = load_chunks("data/processed/apple_2025_chunks.json")
    bm25 = build_bm25_index(chunks)
    vector_store = FAISS.load_local(
        "data/processed/faiss_index",
        embeddings_model,
        allow_dangerous_deserialization=True
    )

    query = "What were Apple's total net sales?"
    results = hybrid_search(query, vector_store, bm25, chunks, top_k=3)

    print(f"Query: {query}\n")
    for i, chunk in enumerate(results):
        print(f"--- Result {i+1} ({chunk['chunk_type']}) ---")
        print(f"Section: {chunk['section_title']}")
        print(f"Text: {chunk['text'][:200]}...\n")