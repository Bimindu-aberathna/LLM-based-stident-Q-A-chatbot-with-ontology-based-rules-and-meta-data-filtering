# Academic Q&A System with Context-Aware Document Retrieval

A research prototype exploring how RAG systems can be enhanced with institutional knowledge for academic environments.

## What's This About?

Traditional chatbots struggle in university settings. Ask them "When's my project deadline?" and they'll give you a generic answer—maybe even the wrong department's rules. This project investigates whether we can do better by teaching the system to understand organizational hierarchies and keep track of what's current.

The core idea: combine semantic search with institutional structure awareness. A Computer Science student shouldn't see Biology department policies just because they use similar words.

## The Research Question

Can we improve retrieval accuracy in institutional knowledge systems by:
1. Modeling organizational hierarchies (department → faculty → university)
2. Prioritizing temporally relevant documents
3. Applying metadata-based filtering rules

Spoiler: Yes, but with interesting tradeoffs.

## Architecture

```
Student Query
    ↓
[Preprocessing & NLP]
    ↓
[Semantic Embedding] → OpenAI text-embedding-3-large
    ↓
[Vector Search] → ChromaDB (HNSW indexing)
    ↓
[Metadata Filtering] → Course codes, departments, validity periods
    ↓
[Hierarchical Prioritization] → Department > Faculty > University
    ↓
[Freshness Scoring] → Recent documents preferred
    ↓
[Context Assembly] → Top-k chunks + neighbor retrieval
    ↓
[LLM Response] → OpenAI GPT-3.5/4
    ↓
Student Answer
```

## Key Components

### Smart Chunking
- Semantic boundary detection using sentence embeddings
- Adaptive chunk sizes (100-450 words)
- 40-word overlaps to preserve context
- Neighbor retrieval for coherence

### Dual Embedding Strategy
- **Preprocessing**: Sentence-BERT (all-MiniLM-L6-v2) for chunk boundaries
- **Retrieval**: OpenAI embeddings for query-document matching
- Trade-off between speed and accuracy

### Ontology-Based Prioritization
Documents get scored on two axes:
- **Hierarchical Authority**: Closer organizational units rank higher
- **Document Freshness**: Recent policies beat outdated ones (usually)

Example: An IT department deadline from last week beats a 2-year-old university-wide policy, even if the old one has better semantic similarity.

### Metadata Filtering
Binary rules prevent irrelevant documents:
- Academic docs: Must match course codes or completed courses
- Non-academic docs: Must match department, faculty, batch, degree program
- Validity periods: Must be currently active

## Technical Stack

**Backend**: FastAPI + Uvicorn  
**Vector DB**: ChromaDB with cosine similarity  
**Embeddings**: OpenAI API + Sentence Transformers  
**NLP**: spaCy for preprocessing  
**LLM**: OpenAI GPT (configurable model)  
**Document Processing**: PyMuPDF, python-docx, PyPDF2  

## Setup

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

```bash
# Clone the repo
git clone https://github.com/Bimindu-aberathna/LLM-based-stident-Q-A-chatbot-with-ontology-based-rules-and-meta-data-filtering.git
cd "RAG Reasearch"

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Set up environment variables
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Running

```bash
# Start the server
uvicorn app.main:app --reload

# Server runs on http://localhost:8000
# API docs at http://localhost:8000/docs
```

## API Endpoints

### Upload Documents

**Academic Document**
```bash
POST /documents/academic_document
Content-Type: multipart/form-data

file: [PDF/DOCX]
course_code: "CS101"
```

**Non-Academic Document**
```bash
POST /documents/non_academic_document
Content-Type: multipart/form-data

file: [PDF/DOCX]
subtype: "policy"
hierarchy_level: "department"
degree_programs: "BSc IT"
departments_affecting: "IT"
faculties_affecting: "Engineering"
batches_affecting: "2021,2022"
validity: "2024-01-01 to 2025-12-31"
```

### Query

```bash
POST /chat/
Content-Type: application/json

{
  "message": "When is the final project deadline?",
  "batch": "2021",
  "department": "Information Technology",
  "degree_program": "BSc (Hons) IT",
  "faculty": "Engineering",
  "current_year": "4",
  "current_sem": "2",
  "specialization": "Software Engineering",
  "course_codes": ["INTE21213", "INTE21333"],
  "courses_done": ["INTE11123", "INTE12234"]
}
```

### Utilities

```bash
GET /documents/debug_chroma          # Inspect database state
DELETE /documents/delete_all          # Clear all documents
POST /documents/clean_string          # Test text preprocessing
POST /documents/chunk_document        # Test chunking algorithm
```

## Project Structure

```
app/
├── main.py                          # FastAPI application
├── routers/
│   ├── chat.py                      # Query endpoint
│   └── documents.py                 # Document management
├── services/
│   ├── document_service.py          # Document processing
│   ├── data_logger.py               # Evaluation logging
│   └── LLM_Services/
│       ├── LLMService_Manager.py    # LLM orchestration
│       └── openai_response_generator.py
├── RAG_Components/
│   ├── chunk_neighbor_retriever.py  # Context expansion
│   ├── chunk_ranker.py              # Hierarchical ranking
│   ├── metadata_filtering_manager.py
│   └── Ontology_ranking.py          # Priority scoring
├── DocumentPreprocessor/
│   ├── text_preprocssor.py          # NLP pipeline
│   ├── pdf_extractor.py
│   └── docx_extractor.py
├── abstract_factory/
│   ├── Database/
│   │   └── chromadb.py              # Vector store
│   └── Embedder/
│       └── open_ai_embedder.py      # Embedding generation
└── models/
    ├── chat.py                      # Request/response models
    └── document.py                  # Document schemas

Documentation/
├── preprocessing.md                  # Text processing methodology
├── rag_llm.md                       # RAG architecture
└── ontology_based_ranking.md       # Prioritization methodology

evaluation/
├── evaluation_responses.csv         # System outputs
└── response_comparison.csv          # Performance metrics
```

## Research Contributions

1. **Institutional Ontology Integration**: First application of hierarchical organizational modeling to RAG retrieval in academic contexts (to our knowledge)

2. **Dual-Factor Prioritization**: Novel combination of authority-based and freshness-based scoring with calibrated weights

3. **Context-Aware Chunking**: Semantic boundary detection using lightweight embeddings during preprocessing

4. **Neighbor Retrieval Strategy**: Automatic context expansion through adjacent chunk retrieval

## Limitations & Future Work

**Current Limitations**:
- Metadata dependency: Garbage in, garbage out
- Cross-cutting policies: Uniform rules (like honor codes) may be under-prioritized
- No multi-turn conversation memory
- English-only support
- Determinism issues in semantic chunking

**Potential Improvements**:
- Multi-turn dialogue with conversation history
- Fine-tuned embeddings on academic corpus
- Active learning for metadata validation
- Policy type classification to handle cross-cutting concerns
- Multilingual support
- Explainability features (show why documents were selected)

## Evaluation

Testing framework logs queries and responses to CSV for human evaluation:
- Ground truth comparison
- Relevance scoring (1-5)
- Accuracy assessment
- Hallucination detection
- Retrieved document validation

See `evaluation/` directory for sample datasets.

## License

MIT License - Academic research project

## Citation

If you use this work, please cite:
```
Aberathna, B. (2025). Ontology-Based Document Prioritization in RAG Systems 
for Academic Information Retrieval. [Research Project].
```

## Acknowledgments

Built with frustration over university chatbots that don't understand departmental policies.

Special thanks to:
- The university administration for generating enough policy documents to make this problem interesting
- OpenAI for the embeddings API
- The ChromaDB team for HNSW indexing that actually works

---

*This is a research prototype. Don't use it to make life-altering academic decisions without verifying the answers.*
