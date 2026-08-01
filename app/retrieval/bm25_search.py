import json
import re
from rank_bm25 import BM25Okapi


def load_chunks(chunks_path: str) -> list[dict]:
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "which", "who",
    "how", "when", "where", "why", "s", "of", "in", "on", "at", "to", "for",
    "and", "or", "but", "with", "by", "from", "this", "that", "it", "as"
}


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r'\b[a-z0-9]+\b', text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def build_bm25_index(chunks: list[dict]):
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def bm25_search(bm25, chunks: list[dict], query: str, k: int = 5) -> list[dict]:
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    results = []
    for idx in ranked_indices[:k]:
        results.append({
            "chunk": chunks[idx],
            "score": scores[idx]
        })
    return results


if __name__ == "__main__":
    chunks = load_chunks("data/processed/apple_2025_chunks.json")
    bm25 = build_bm25_index(chunks)
    
    query = "What were Apple's total net sales?"
    results = bm25_search(bm25, chunks, query, k=3)
    
    print(f"Query: {query}\n")
    for i, result in enumerate(results):
        print(f"--- Result {i+1} (score: {result['score']:.2f}) ---")
        print(f"Section: {result['chunk']['section_title']}")
        print(f"Text: {result['chunk']['text'][:200]}...\n")