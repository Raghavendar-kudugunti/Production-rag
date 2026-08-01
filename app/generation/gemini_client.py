import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_hypothetical_questions(chunk_text: str, num_questions: int = 3) -> list[str]:
    prompt = f"""Given the following excerpt from a financial document (10-K filing), generate {num_questions} specific questions that this text directly answers. 

Rules:
- Questions must be answerable using ONLY the information in this excerpt
- Be specific (mention actual figures, terms, or facts from the text where relevant)
- Return ONLY the questions, one per line, no numbering, no extra text

Excerpt:
{chunk_text}
"""
    
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )
    
    questions = [q.strip() for q in response.text.strip().split("\n") if q.strip()]
    return questions


if __name__ == "__main__":
    test_chunk = "The Company's total net sales for fiscal 2025 were $391.0 billion, an increase of 6% compared to fiscal 2024, driven primarily by growth in Services and iPhone revenue."
    
    questions = generate_hypothetical_questions(test_chunk)
    print(f"Generated {len(questions)} questions:\n")
    for q in questions:
        print(f"- {q}")