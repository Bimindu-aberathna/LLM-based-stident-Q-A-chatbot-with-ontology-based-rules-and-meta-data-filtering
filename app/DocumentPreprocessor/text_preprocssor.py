import re
from typing import List
import spacy
from sentence_transformers import SentenceTransformer, util


class TextPreprocessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.stopwords = set(spacy.lang.en.stop_words.STOP_WORDS)

    def clean_text(self, text: str) -> str:
        # Remove headers/footers patterns
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove excessive whitespace but preserve paragraph structure
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove excessive punctuation but preserve sentence structure
        text = re.sub(r'[^\w\s\.\?\!\,\;\:\-\(\)]', ' ', text)
        text = re.sub(r'\.{2,}', '...', text)
        text = self.case_normalize(text)
        return text.strip()   
    
    def case_normalize(self, text: str) -> str:
        #lowercase text
        return text.lower() 
    
    def nlp_process(self, text: str) -> str:
        # Process text with spaCy
        doc = self.nlp(text)
        processed_tokens = []
        for token in doc:
            if not token.is_punct and not token.is_space and token.text.lower() not in self.stopwords:
                processed_tokens.append(token.lemma_.lower())
        
        return ' '.join(processed_tokens)
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        if overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk_size")
        
        words = text.split()
        if not words:
            return []
        
        chunks = []
        step = chunk_size - overlap
        
        for i in range(0, len(words), step):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            # Stop if we've reached the end
            if i + chunk_size >= len(words):
                break
        
        return chunks

    def smart_chunk_text(self, text: str, max_chunk_tokens: int = 500, min_chunk_tokens: int = 100, similarity_threshold: float = 0.65) -> list[str]:
        # Load a sentence embedding model (fast & lightweight)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Step 1: Pre-clean text
        clean_text = re.sub(r'\s+', ' ', text.strip())
        
        # Step 2: Split into sentences (basic rule-based)
        sentences = re.split(r'(?<=[.!?])\s+', clean_text)
        if not sentences:
            return []
        
        chunks, current_chunk = [], []
        current_chunk_len = 0
        
        # Step 3: Embed sentences
        embeddings = model.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
        
        # Step 4: Group semantically similar sentences into chunks
        for i, sentence in enumerate(sentences):
            token_len = len(sentence.split())
            
            # If adding sentence exceeds max length → start new chunk
            if current_chunk_len + token_len > max_chunk_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_chunk_len = token_len
                continue
            
            # If semantically dissimilar → start new chunk
            if current_chunk and util.cos_sim(embeddings[i], embeddings[i-1]) < similarity_threshold:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_chunk_len = token_len
            else:
                # Add to current chunk
                current_chunk.append(sentence)
                current_chunk_len += token_len
        
        # Step 5: Add final chunk
        if current_chunk:
            if len(" ".join(current_chunk).split()) >= min_chunk_tokens:
                chunks.append(" ".join(current_chunk))

        processed_chunks = self.enforce_min_size(chunks, min_chunk_tokens)
        return self.add_overlap(processed_chunks, overlap_tokens=50)

    def enforce_min_size(self, chunks: List[str], min_chunk_tokens: int = 50) -> List[str]:
        if not chunks:
            return []

        fixed_chunks = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i].strip()
            word_count = len(chunk.split())

            # If chunk is too small, merge with neighbor
            if word_count < min_chunk_tokens:
                if fixed_chunks:  # merge with previous if exists
                    fixed_chunks[-1] = fixed_chunks[-1].strip() + " " + chunk
                elif i + 1 < len(chunks):  # otherwise merge with next
                    chunks[i + 1] = chunk + " " + chunks[i + 1]
                else:  # only one very short chunk
                    fixed_chunks.append(chunk)
            else:
                fixed_chunks.append(chunk)
            i += 1

        return fixed_chunks
    
    def add_overlap(self, chunks, overlap_tokens=50) -> List[str]:
        if overlap_tokens <= 0:
            return chunks  
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            words = chunk.split()
            if i > 0 and overlap_tokens > 0:
                # prepend last N words of previous chunk
                prev_words = chunks[i-1].split()
                prefix = prev_words[-overlap_tokens:] if len(prev_words) > overlap_tokens else prev_words
                merged = " ".join(prefix + words)
                overlapped_chunks.append(merged)
            else:
                overlapped_chunks.append(chunk)
        
        return overlapped_chunks