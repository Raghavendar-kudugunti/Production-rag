import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from google import genai

from retrieval.bm25_search import load_chunks, build_bm25_index
from retrieval.hybrid_search import hybrid_search, vector_search

load_dotenv()

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

TEST_QUESTIONS = [
    {"question": "What were Apple's total net sales for fiscal 2025?", "expected_answer": "$416,161 million"},
    {"question": "How much did Apple spend on research and development in fiscal 2025?", "expected_answer": "$34,550 million"},
    {"question": "What was Apple's net income for fiscal 2025?", "expected_answer": "$112,010 million"},
    {"question": "What was Apple's gross margin for fiscal 2025?", "expected_answer": "$195,201 million"},
    {"question": "What were Apple's total operating expenses in fiscal 2025?", "expected_answer": "$62,151 million"},
    {"question": "What was Apple's diluted earnings per share for fiscal 2025?", "expected_answer": "$7.46"},
    {"question": "How much did net sales grow in Europe during fiscal 2025?", "expected_answer": "10%"},
    {"question": "What percentage did Services net sales grow in fiscal 2025?", "expected_answer": "14%"},
    {"question": "What is Apple's Item 1A Risk Factors section primarily about?", "expected_answer": "material risks to business, reputation, operations, financial condition and stock price"},
    {"question": "Does Apple face risks related to cybersecurity?", "expected_answer": "yes, Item 1C addresses cybersecurity risks"},
    {"question": "What was Apple's total cost of sales in fiscal 2025?", "expected_answer": "$220,960 million"},
    {"question": "How much did Apple's net sales grow in Japan during fiscal 2025?", "expected_answer": "15%"},
]


def call_gemini_with_retry(prompt: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait_time = 20 * (attempt + 1)
                print(f"    Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    return ""


def naive_search(query: str, k: int = 5) -> list[dict]:
    chunk_ids = vector_search(vector_store, query, k=k)
    chunk_lookup = {c["chunk_id"]: c for c in chunks}
    return [chunk_lookup[cid] for cid in chunk_ids if cid in chunk_lookup]


def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {c['section_title']}]\n{c['text']}" for c in retrieved_chunks
    )
    prompt = f"""You are a financial analyst assistant. Answer the question using ONLY the context below, but don't just copy fragments — synthesize a clear, well-explained answer in your own words, as an analyst would explain it to someone.

Include the specific figures/facts, but also brief context (e.g., what drove a change, what period it covers) if the source material supports it. If the answer isn't in the context, say so clearly rather than guessing.

Context:
{context}

Question: {question}

Answer:"""

    return call_gemini_with_retry(prompt)


def generate_answer_no_retrieval(question: str) -> str:
    prompt = f"""Answer this question about Apple's fiscal 2025 10-K filing based on your own knowledge.

Question: {question}

Answer:"""
    return call_gemini_with_retry(prompt)


def judge_answer(question: str, expected: str, actual: str) -> dict:
    prompt = f"""You are evaluating an AI system's answer for correctness.

Question: {question}
Expected answer (key facts): {expected}
Actual answer: {actual}

Does the actual answer contain the correct key facts from the expected answer? Respond in this exact format:
SCORE: [1 if correct, 0 if incorrect]
REASON: [one sentence explanation]"""

    text = call_gemini_with_retry(prompt)
    score = 1 if "SCORE: 1" in text else 0
    reason = text.split("REASON:")[-1].strip() if "REASON:" in text else ""
    return {"score": score, "reason": reason}


if __name__ == "__main__":
    chunks = load_chunks("data/processed/apple_2025_chunks.json")
    bm25 = build_bm25_index(chunks)
    vector_store = FAISS.load_local(
        "data/processed/faiss_index",
        embeddings_model,
        allow_dangerous_deserialization=True
    )

    results = []
    naive_correct = 0
    hybrid_correct = 0
    no_rag_correct = 0
    naive_times = []
    hybrid_times = []
    no_rag_times = []

    for i, item in enumerate(TEST_QUESTIONS):
        question = item["question"]
        expected = item["expected_answer"]
        print(f"\n[{i+1}/{len(TEST_QUESTIONS)}] {question}")

        t0 = time.time()
        naive_chunks = naive_search(question, k=5)
        naive_answer = generate_answer(question, naive_chunks)
        naive_time = time.time() - t0
        naive_times.append(naive_time)
        naive_judgment = judge_answer(question, expected, naive_answer)
        naive_correct += naive_judgment["score"]
        print(f"  Naive  ({naive_time:.1f}s): {'✅' if naive_judgment['score'] else '❌'} {naive_answer[:90]}")

        t0 = time.time()
        hybrid_chunks = hybrid_search(question, vector_store, bm25, chunks, top_k=5)
        hybrid_answer = generate_answer(question, hybrid_chunks)
        hybrid_time = time.time() - t0
        hybrid_times.append(hybrid_time)
        hybrid_judgment = judge_answer(question, expected, hybrid_answer)
        hybrid_correct += hybrid_judgment["score"]
        print(f"  Hybrid ({hybrid_time:.1f}s): {'✅' if hybrid_judgment['score'] else '❌'} {hybrid_answer[:90]}")

        t0 = time.time()
        no_rag_answer = generate_answer_no_retrieval(question)
        no_rag_time = time.time() - t0
        no_rag_times.append(no_rag_time)
        no_rag_judgment = judge_answer(question, expected, no_rag_answer)
        no_rag_correct += no_rag_judgment["score"]
        print(f"  No-RAG ({no_rag_time:.1f}s): {'✅' if no_rag_judgment['score'] else '❌'} {no_rag_answer[:90]}")

        results.append({
            "question": question,
            "expected": expected,
            "naive_answer": naive_answer,
            "naive_score": naive_judgment["score"],
            "naive_time_sec": round(naive_time, 2),
            "hybrid_answer": hybrid_answer,
            "hybrid_score": hybrid_judgment["score"],
            "hybrid_time_sec": round(hybrid_time, 2),
            "no_rag_answer": no_rag_answer,
            "no_rag_score": no_rag_judgment["score"],
            "no_rag_time_sec": round(no_rag_time, 2),
        })

        with open("data/processed/eval_results.json", "w", encoding="utf-8") as f:
            json.dump({
                "naive_accuracy": naive_correct / (i + 1),
                "hybrid_accuracy": hybrid_correct / (i + 1),
                "no_rag_accuracy": no_rag_correct / (i + 1),
                "avg_naive_time_sec": round(sum(naive_times) / len(naive_times), 2),
                "avg_hybrid_time_sec": round(sum(hybrid_times) / len(hybrid_times), 2),
                "avg_no_rag_time_sec": round(sum(no_rag_times) / len(no_rag_times), 2),
                "results": results
            }, f, indent=2, ensure_ascii=False)

        time.sleep(5)

    total = len(TEST_QUESTIONS)
    print(f"\n{'='*60}")
    print(f"{'Method':<12} {'Accuracy':<18} {'Avg Latency':<12}")
    print(f"{'-'*60}")
    print(f"{'No-RAG':<12} {no_rag_correct}/{total} ({100*no_rag_correct/total:.1f}%)   {sum(no_rag_times)/total:.1f}s")
    print(f"{'Naive RAG':<12} {naive_correct}/{total} ({100*naive_correct/total:.1f}%)   {sum(naive_times)/total:.1f}s")
    print(f"{'Hybrid RAG':<12} {hybrid_correct}/{total} ({100*hybrid_correct/total:.1f}%)   {sum(hybrid_times)/total:.1f}s")
    print(f"\nSaved detailed results to data/processed/eval_results.json")