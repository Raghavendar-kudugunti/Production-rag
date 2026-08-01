import re
import tiktoken

encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(encoder.encode(text))


def split_into_sentences(text: str) -> list[str]:
    sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_endings, text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_section(section_text: str, max_tokens: int = 450, overlap_tokens: int = 50) -> list[str]:
    sentences = split_into_sentences(section_text)
    
    chunks = []
    current_chunk_sentences = []
    current_token_count = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        
        if current_token_count + sentence_tokens > max_tokens and current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
            
            overlap_sentences = []
            overlap_count = 0
            for s in reversed(current_chunk_sentences):
                s_tokens = count_tokens(s)
                if overlap_count + s_tokens > overlap_tokens:
                    break
                overlap_sentences.insert(0, s)
                overlap_count += s_tokens
            
            current_chunk_sentences = overlap_sentences
            current_token_count = overlap_count
        
        current_chunk_sentences.append(sentence)
        current_token_count += sentence_tokens
    
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
    
    return chunks

if __name__ == "__main__":
    from pdf_loader import extract_text_from_pdf, detect_item_sections
    
    text = extract_text_from_pdf("data/raw/aapl-20250927.pdf")
    sections = detect_item_sections(text)
    
    risk_factors = next(s for s in sections if "Risk Factors" in s["title"])
    chunks = chunk_section(risk_factors["text"])
    
    print("End of Chunk 1 (last 150 chars):")
    print(chunks[0][-150:])
    print("\nStart of Chunk 2 (first 150 chars):")
    print(chunks[1][:150])