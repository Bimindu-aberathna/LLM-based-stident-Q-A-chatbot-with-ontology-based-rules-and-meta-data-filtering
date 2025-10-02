import re
from typing import List, Optional
import spacy
from sentence_transformers import SentenceTransformer, util


class TextPreprocessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.stopwords = set(spacy.lang.en.stop_words.STOP_WORDS)
        self._embed_model: Optional[SentenceTransformer] = None  # lazy load

    def _get_embed_model(self):
        if self._embed_model is None:
            self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embed_model
    def nlp_process(self, text: str) -> str:
        # Process text with spaCy
        critical_stopwords = {
            "not", "no", "nor", "never", "n't", "none", "nothing", "nowhere", "neither", 
            "hardly", "scarcely", "barely", "will", "be", "during", "despite", "although", 
            "however", "accordingly", "initially", "but", "yet", "still", "even", "also",
            "only", "just", "all", "any", "each", "every", "both", "either", "would",
            "should", "could", "must", "may", "might", "can", "shall", "have", "has", "had"
        }
        
        # Basic stopwords to remove
        basic_stopwords = {
            "the", "a", "an", "and", "in", "on", "at", "to", "for", "of", "by", "about", "into", "through", "during", "above", 
            "below", "between", "among", "this", "that", "these", "those", "i", "me", "my",
            "myself", "we", "our", "ours", "ourselves",  "yourself",
            "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself",
            "it", "its", "itself", "they", "them", "their", "theirs", "themselves"
        }
        
        # Remove basic stopwords but keep critical ones
        self.stopwords = basic_stopwords
        
        doc = self.nlp(text)
        processed_tokens = []
        for token in doc:
            if not token.is_punct and not token.is_space:
                token_lower = token.text.lower()
                # Keep critical stopwords or non-stopwords
                if token_lower in critical_stopwords or token_lower not in self.stopwords:
                    processed_tokens.append(token.lemma_.lower())
        
        return ' '.join(processed_tokens)
    

    def clean_text(self, text: str) -> str:
        text = re.sub(r'Page \d+ of \d+', '', text) 
        text = re.sub(r'\n\s*\n', '\n\n', text) 
        text = re.sub(r'[ \t]+', ' ', text) 
        text = re.sub(r'\.{2,}', '...', text) 
        return text.strip()

    def smart_chunk_text(
        self,
        text: str,
        max_chunk_words: int = 450,
        min_chunk_words: int = 100,
        similarity_threshold: float = 0.75,
        overlap_words: int = 40
    ) -> List[str]:
        """
        Semantic chunking with strict size guarantees.

        Guarantees:
          - Base chunk length (before overlap) <= max_chunk_words
          - Overlapped chunk length <= max_chunk_words + overlap_words
          - Only final tiny remainder may be < min_chunk_words (kept to avoid data loss)
        """
        cleaned = self.clean_text(text)
        if not cleaned:
            return []

        # Sentence segmentation via spaCy (robust vs regex + lowercase issues)
        doc = self.nlp(cleaned)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
        if not sentences:
            return []

        # Split ultra-long sentences first
        processed_sentences: List[str] = []
        for sent in sentences:
            words = sent.split()
            if len(words) <= max_chunk_words:
                processed_sentences.append(sent)
            else:
                # Clause-level split by punctuation, then enforce cap
                clauses = re.split(r'[,;:](?=\s)', sent)
                buffer = []
                buf_len = 0
                for clause in clauses:
                    w = clause.strip().split()
                    if buf_len + len(w) <= max_chunk_words:
                        buffer.extend(w)
                        buf_len += len(w)
                    else:
                        if buffer:
                            processed_sentences.append(" ".join(buffer))
                        buffer = w[:max_chunk_words]  # enforce cap
                        buf_len = len(buffer)
                        # If remaining clause still too long, chunk it
                        if len(w) > max_chunk_words:
                            tail = w[max_chunk_words:]
                            while tail:
                                processed_sentences.append(" ".join(tail[:max_chunk_words]))
                                tail = tail[max_chunk_words:]
                            buffer = []
                            buf_len = 0
                if buffer:
                    processed_sentences.append(" ".join(buffer))

        if not processed_sentences:
            return []

        # Embeddings for semantic boundary detection
        model = self._get_embed_model()
        embeddings = model.encode(
            processed_sentences,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        base_chunks: List[List[str]] = []
        current: List[str] = []
        current_len = 0

        for i, sent in enumerate(processed_sentences):
            sent_words = sent.split()
            sw_len = len(sent_words)

            # If adding exceeds size cap → flush
            if current and current_len + sw_len > max_chunk_words:
                base_chunks.append(current)
                current = sent_words
                current_len = sw_len
                continue

            # Semantic boundary?
            semantic_break = False
            if current and i > 0:
                sim = float(util.cos_sim(embeddings[i], embeddings[i - 1]))
                semantic_break = sim < similarity_threshold

            if semantic_break and current_len >= min_chunk_words:
                base_chunks.append(current)
                current = sent_words
                current_len = sw_len
            else:
                current.extend(sent_words)
                current_len += sw_len

            # Hard safety (shouldn't trigger normally)
            if current_len > max_chunk_words:
                base_chunks.append(current[:max_chunk_words])
                leftover = current[max_chunk_words:]
                current = leftover
                current_len = len(leftover)

        if current:
            base_chunks.append(current)

        # Merge ONLY a trailing tiny chunk (< min) if possible
        if len(base_chunks) >= 2 and len(base_chunks[-1]) < min_chunk_words:
            tail = base_chunks.pop()
            prev = base_chunks[-1]
            if len(prev) + len(tail) <= max_chunk_words:
                base_chunks[-1] = prev + tail
            else:
                # Keep tail as its own small chunk (better than overflow)
                base_chunks.append(tail)

        # Convert to text
        base_texts = [" ".join(words) for words in base_chunks]

        # Add overlap
        if overlap_words > 0 and len(base_texts) > 1:
            final_chunks: List[str] = []
            for idx, chunk_text in enumerate(base_texts):
                if idx == 0:
                    final_chunks.append(chunk_text)
                    continue
                prev_words = base_texts[idx - 1].split()
                overlap = prev_words[-overlap_words:] if len(prev_words) > overlap_words else prev_words
                combined_words = overlap + chunk_text.split()
                if len(combined_words) > max_chunk_words + overlap_words:
                    combined_words = combined_words[:max_chunk_words + overlap_words]
                final_chunks.append(" ".join(combined_words))
        else:
            final_chunks = base_texts

        # Debug (word counts)
        for i, ch in enumerate(final_chunks, 1):
            wc = len(ch.split())
            print(f"😎😎[Chunk {i}]: {wc} words😎😎")

        return final_chunks