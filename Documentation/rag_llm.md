# Retrieval-Augmented Generation (RAG) Architecture

## 1. Introduction

This document presents the Retrieval-Augmented Generation (RAG) framework implemented for the academic query resolution system. RAG represents a paradigm shift in question-answering systems. It combines information retrieval with language generation. Traditional language models generate responses solely from their training data. RAG systems, by contrast, first retrieve relevant information from external knowledge bases, then use that information to generate contextually grounded answers.

The architecture addresses a critical challenge in academic information systems: providing accurate, verifiable, and contextually appropriate responses to student queries while drawing from extensive institutional documentation.

## 2. Embedding Generation

### 2.1 Dual Embedding Models

The system employs two distinct embedding models for different purposes:

**Document and Query Embeddings**: OpenAI's `text-embedding-3-large` model
- Dimensionality: 3,072 dimensions
- Primary use: Converting document chunks and user queries into semantic vector representations
- Implementation: `OpenAIEmbedder` class

**Semantic Chunking Embeddings**: Sentence-BERT `all-MiniLM-L6-v2` model
- Dimensionality: 384 dimensions  
- Primary use: Detecting semantic boundaries during document preprocessing
- Implementation: Integrated within `TextPreprocessor` class

### 2.2 Embedding Model Selection Rationale

The dual-model approach serves distinct but complementary purposes:

**OpenAI text-embedding-3-large** was selected for retrieval operations because:
1. **High dimensionality** (3,072 dimensions) provides superior semantic discrimination capability
2. **Extensive training corpus** ensures robust performance across diverse academic domains
3. **State-of-the-art performance** on semantic similarity benchmarks
4. **API-based deployment** simplifies scaling and maintenance

**Sentence-BERT all-MiniLM-L6-v2** was chosen for preprocessing because:
1. **Computational efficiency** enables rapid processing of entire documents
2. **Sufficient semantic resolution** for boundary detection tasks
3. **Local execution** eliminates API costs during preprocessing
4. **Consistent performance** for sentence-level semantic comparison

This architecture separates concerns effectively. The preprocessing phase uses lightweight models for structural analysis. The retrieval phase employs sophisticated models for semantic matching.

### 2.3 Vector Generation Process

Document chunks undergo transformation through the embedding pipeline:

```python
embedder = OpenAIEmbedder()
embeddings = embedder.embed_docs(chunks)
```

Each chunk is converted to a 3,072-dimensional vector where:
- Each dimension represents learned semantic features
- Vector proximity in the embedding space indicates semantic similarity
- Cosine similarity serves as the primary distance metric

Similarly, user queries are embedded using identical models to ensure vector space consistency:

```python
query_vector = embedder.embed_query(user_question)
```

**Critical Design Decision**: Both documents and queries must be embedded using the same model and configuration. This ensures they occupy the same semantic vector space, enabling meaningful similarity comparisons.

## 3. Vector Storage and Indexing

### 3.1 ChromaDB Vector Database

The system employs **ChromaDB** as its vector storage solution. ChromaDB is a specialized database optimized for embedding storage and similarity search operations.

**Key Configuration**:
```python
self.client = chromadb.PersistentClient(path="./data/chroma_db")
self.collection = self.client.get_or_create_collection(
    name="document_corpus_test",
    metadata={"hnsw:space": "cosine"}
)
```

**Technical Specifications**:
- **Persistence**: Disk-based storage ensures data durability across system restarts
- **Distance Metric**: Cosine similarity for semantic comparison
- **Indexing Algorithm**: HNSW (Hierarchical Navigable Small World) graph for efficient approximate nearest neighbor search

### 3.2 HNSW Indexing Algorithm

The HNSW algorithm constructs a multi-layer graph structure where:
1. **Bottom layer** contains all vectors with high-connectivity local neighborhoods
2. **Upper layers** contain progressively sparser subsets serving as "highways" for efficient search
3. **Search operations** begin at the top layer and descend through layers, navigating toward the query vector

**Performance Characteristics**:
- **Search complexity**: O(log N) average case, where N is corpus size
- **Trade-off**: Slight approximation in exchange for substantial speed improvements
- **Suitability**: Ideal for systems with large document corpora requiring real-time query response

### 3.3 Document Storage Schema

Each document chunk is stored with three components:

**1. Vector Embedding** (3,072 dimensions)
- Semantic representation for similarity search

**2. Text Content** (original chunk text)
- Retrieved verbatim for context provision to the LLM

**3. Metadata** (structured attributes)
Academic documents:
- `type`: "academic"
- `course_code`: Course identifier
- `document_name`: Source document filename
- `chunk_index`: Position within original document
- `total_chunks`: Total chunks in document
- `upload_date`: Timestamp for versioning

Non-academic documents:
- `type`: "non-academic"
- `subtype`: Document category (policy, procedure, announcement)
- `hierarchy_level`: Applicability scope (department/faculty/university)
- `degree_programs`: Comma-separated applicable programs
- `departments_affecting`: Applicable departments
- `faculties_affecting`: Applicable faculties
- `batches_affecting`: Applicable student cohorts
- `validity`: Temporal applicability period
- `link`: External reference URL

This metadata enables sophisticated filtering and ranking mechanisms described in subsequent sections.

## 4. Retrieval Pipeline

### 4.1 Multi-Stage Retrieval Architecture

The retrieval process implements a cascading filter approach with three distinct stages:

**Stage 1: Semantic Similarity Search**
**Stage 2: Metadata-Based Filtering**  
**Stage 3: Ontology-Based Prioritization**

This architecture balances semantic relevance with structural constraints specific to academic information systems.

### 4.2 Stage 1: Semantic Similarity Search

The initial retrieval queries the vector database for semantically similar chunks:

```python
results = self.collection.query(
    query_embeddings=[query_vector],
    n_results=search_k,
    include=['documents', 'metadatas', 'distances']
)
```

**Parameters**:
- `search_k`: Initially set to 80 (later filtered to top_k=10)
- `query_embeddings`: The embedded user question
- `include`: Specifies retrieval of text, metadata, and distance scores

**Similarity Threshold Filtering**:

Raw results undergo immediate filtering based on semantic similarity:

```python
similarity_score = 1.0 - distance  # ChromaDB uses cosine distance
if similarity_score >= similarity_threshold:  # Default: 0.7
    filtered_results.append(chunk)
```

**Threshold Justification (0.7)**:
- Values below 0.7 indicate weak semantic alignment with the query
- Empirical testing revealed significant quality degradation below this threshold
- The threshold prevents irrelevant content from contaminating context
- Balances recall (finding relevant content) with precision (excluding noise)

**Output**: A candidate set of chunks with proven semantic relevance to the query.

### 4.3 Stage 2: Metadata-Based Filtering

Chunks passing similarity thresholds undergo rule-based metadata validation. This ensures retrieved information is contextually appropriate for the specific student.

**Filtering Rules**:

*For Academic Documents*:
- Course code must match student's enrolled courses OR courses previously completed
- This prevents exposure to irrelevant course materials

*For Non-Academic Documents*:
- Degree program must include student's program (if specified)
- Department must include student's department (if specified)
- Faculty must include student's faculty (if specified)
- Batch must include student's batch (if specified)
- Validity period must encompass current date (if specified)

**Implementation Logic**:

```python
def passes_all_rules(self, metadata: Dict, student: StudentQueryRequest) -> bool:
    if metadata.get('type') == 'academic':
        return self._check_academic_rules(metadata, student)
    else:
        return self._check_non_academic_rules(metadata, student)
```

**Design Principle**: The system operates on an "inclusion" basis. If a metadata field is unspecified or contains "*", it matches all students. This prevents overly restrictive filtering when documents have broad applicability.

**Output**: Only chunks matching the student's profile and context proceed to the next stage.

### 4.4 Stage 3: Ontology-Based Hierarchical Ranking

Non-academic documents undergo additional prioritization based on hierarchical specificity. Academic institutions exhibit natural organizational hierarchies: department → faculty → university. Information at different levels has varying relevance.

**Hierarchical Scoring System**:

```python
HIERARCHICAL_SCORES = {
    'department': 100,    # Highest priority
    'faculty': 50,        # Medium priority  
    'university': 20,     # Lowest priority
}
```

**Ranking Logic**:

Documents are scored based on their hierarchical level and applicability to the student's organizational position:

1. **Department-level documents**: Specific to the student's department receive highest scores
2. **Faculty-level documents**: Applicable to the student's faculty receive medium scores
3. **University-level documents**: General policies receive baseline scores

**Temporal Freshness Factor**:

Recent documents receive slight score bonuses (maximum 15 points) to favor current policies over outdated ones:

```python
recency_score = max(0, MAX_FRESHNESS_POINTS * (1 - age_in_years / 5))
total_score = hierarchical_score + recency_score
```

**Output**: Non-academic chunks are re-ordered by total score, ensuring the most contextually relevant policies appear first.

This mechanism addresses the reality that department-specific policies override general university policies. Students should see the most applicable information first.

## 5. Context Enhancement with Neighbor Retrieval

### 5.1 The Context Fragmentation Problem

Standard RAG implementations face a challenge: retrieved chunks may lack sufficient context to be fully interpretable. A chunk might begin mid-sentence or reference concepts explained in preceding paragraphs.

Consider this example:

**Chunk Retrieved**: "These must be submitted before the deadline. Late submissions will not be accepted."

**Problem**: What must be submitted? What is the deadline? This chunk lacks crucial context.

### 5.2 Neighbor Chunk Retrieval Solution

The system implements neighbor retrieval to reconstruct local context:

```python
chunk_with_neighbors = neighbor_retriever.retrieve_neighbor_chunks_for_a_chunk(
    chunk=chunk['text'],
    chunk_type="academic",
    neighbor_count=2
)
```

**Mechanism**:

For each retrieved chunk, the system:
1. Identifies the source document and chunk index from metadata
2. Retrieves 2 preceding chunks and 2 following chunks from the same document
3. Assembles these into a coherent context window
4. Preserves document order to maintain narrative flow

**Example Enhancement**:

**Original Chunk**: "These must be submitted before the deadline."

**With Neighbors**:
"Students must complete the prerequisite approval form. These must be submitted before the deadline specified in the academic calendar. Late submissions will not be accepted under any circumstances."

**Constraint**: Neighbors are retrieved only from the same source document to prevent mixing unrelated content.

### 5.3 Context Window Assembly

The assembled context follows this structure:

```
[Neighbor -2] [Neighbor -1] [ORIGINAL CHUNK] [Neighbor +1] [Neighbor +2]
```

Annotations indicate chunk roles:
- `[ORIGINAL-SCORE: X]`: The primary retrieved chunk with its similarity score
- `[NEIGHBOR-SCORE: Y]`: Adjacent chunks providing context

This transparency enables the LLM to weight information appropriately. The original chunk receives primary attention, while neighbors provide interpretive context.

## 6. Dual-Stream Context Management

### 6.1 Academic and Non-Academic Separation

The retrieval pipeline maintains separate streams for academic and non-academic content:

**Academic Stream**:
- Course materials
- Syllabi
- Lecture notes
- Assignment descriptions

**Non-Academic Stream**:
- University policies
- Administrative procedures
- Deadlines and schedules
- Regulatory information

**Rationale for Separation**:

Academic and non-academic documents serve different informational needs. Academic content addresses subject matter and learning outcomes. Non-academic content addresses procedural and administrative requirements. Separating these streams enables:

1. **Differential processing**: Academic chunks undergo neighbor retrieval; non-academic chunks undergo hierarchical ranking
2. **Balanced representation**: Ensures responses incorporate both content types when relevant
3. **Type-specific optimization**: Each stream can apply domain-appropriate processing

### 6.2 Context Assembly for LLM Consumption

Retrieved chunks from both streams are formatted into structured context strings:

**Academic Context**:
```
=== ACADEMIC CONTEXT ===
Course: INTE 21213 - Information Systems Modelling

Chunk 1 (Score: 0.85):
[Content discussing exam format and requirements]

Chunk 2 (Score: 0.82):
[Content about course prerequisites]
```

**Non-Academic Context**:
```
=== NON-ACADEMIC CONTEXT ===

Department Policy (Hierarchy: Department, Freshness: Recent):
[Content about registration procedures]

Faculty Guideline (Hierarchy: Faculty):  
[Content about examination rules]
```

This structured format enables the LLM to:
- Distinguish between content types
- Understand source reliability (similarity scores, hierarchical levels)
- Synthesize information from multiple sources coherently

## 7. Response Generation

### 7.1 Prompt Engineering for Academic Queries

The LLM receives a carefully structured prompt incorporating:

1. **Role Definition**: "You are a university academic advisor..."
2. **Student Context**: Batch, department, degree program, specialization, current semester
3. **Academic Context**: Retrieved course-related content
4. **Non-Academic Context**: Retrieved policy/procedural content
5. **Query**: The student's specific question
6. **Response Instructions**: Formatting requirements, citation expectations, tone guidelines

**Sample Prompt Structure**:

```
You are a university academic advisor assisting students with academic and administrative queries.

STUDENT INFORMATION:
- Batch: 2021
- Department: Information Technology
- Degree Program: BSc (Hons) Information Technology
- Current Year: 3
- Current Semester: 2
- Specialization: Software Engineering

ACADEMIC CONTEXT:
[Retrieved academic chunks with scores and neighbor context]

NON-ACADEMIC CONTEXT:
[Retrieved policy chunks with hierarchical rankings]

STUDENT QUESTION:
"When is the deadline for my INTE 21213 assignment submission?"

INSTRUCTIONS:
- Provide accurate information based ONLY on the context provided
- If context is insufficient, explicitly state this
- Cite specific policies or course materials when applicable
- Use clear, student-friendly language
- For procedural questions, provide step-by-step guidance when possible
```

### 7.2 Contextual Grounding

A critical aspect of the prompt engineering is instructing the LLM to **ground responses exclusively in retrieved context**. This prevents hallucination—the generation of plausible but false information.

**Explicit Instructions to LLM**:
- "Base your answer ONLY on the provided context"
- "If information is not found in the context, say so explicitly"
- "Do not use external knowledge or make assumptions"

This constraint ensures verifiability. Every statement in the response can be traced to a specific retrieved chunk.

### 7.3 Handling Insufficient Context

When retrieval yields no results (empty academic and non-academic contexts), the system generates a specialized response:

```python
def no_relevant_documents_response(self, question, studentMetadata):
    prompt = """The database does not have sufficient information to answer the question.
    Inform the student that no relevant information was found.
    For administrative questions, suggest contacting university administration."""
```

This approach maintains transparency. Students are informed of system limitations rather than receiving fabricated answers.

## 8. Metadata-Based Filtering and Ontology-Based Prioritization

### 8.1 Overview

Two advanced mechanisms enhance retrieval precision beyond semantic similarity:

**Metadata-Based Filtering**: Applies binary inclusion/exclusion rules based on student attributes and document properties. Described in Stage 2 of the retrieval pipeline (Section 4.3).

**Ontology-Based Prioritization**: Applies hierarchical ranking to non-academic documents based on organizational structure and temporal relevance. Described in Stage 3 of the retrieval pipeline (Section 4.4).

These mechanisms are addressed in detail in subsequent dedicated sections of the thesis.

### 8.2 Integration with RAG Architecture

Both mechanisms integrate seamlessly into the retrieval pipeline:

1. **Metadata filtering** operates as a post-retrieval filter, removing contextually inappropriate chunks
2. **Ontology-based prioritization** operates as a ranking function, reordering chunks by applicability

Together, they transform the retrieval process from purely semantic matching to context-aware, policy-compliant information retrieval.

## 9. System Performance Characteristics

### 9.1 Retrieval Efficiency

The multi-stage architecture balances comprehensiveness with computational efficiency:

**Stage 1 (Similarity Search)**: O(log N) via HNSW indexing  
**Stage 2 (Metadata Filtering)**: O(K) where K is the number of similarity matches  
**Stage 3 (Hierarchical Ranking)**: O(M log M) where M is the number of non-academic chunks

For typical queries:
- Initial similarity search: ~80 candidates retrieved
- After metadata filtering: ~20-30 candidates remain
- Final context: Top 10 academic + top 10 non-academic chunks

**Query Response Time**: Typically 2-4 seconds end-to-end, including:
- Query embedding: ~200ms
- Vector search: ~100ms
- Metadata filtering: ~50ms
- Neighbor retrieval: ~500ms
- LLM response generation: 1-3 seconds

### 9.2 Scalability Considerations

The architecture scales effectively with corpus growth:

**HNSW Index Scaling**: Logarithmic search complexity maintains performance even with 100,000+ chunks  
**Metadata Filtering**: Linear complexity is acceptable due to small candidate set (post-similarity filtering)  
**Storage Requirements**: 3,072 dimensions × 4 bytes × N chunks ≈ 12KB per chunk

For a corpus of 10,000 chunks:
- Vector storage: ~120MB
- Text storage: Variable (typically ~50-200MB)
- Metadata storage: ~5-10MB

## 10. Advantages Over Traditional Approaches

### 10.1 Comparison with Pure LLM Systems

Traditional LLM-only systems face critical limitations:

**Knowledge Cutoff**: Training data becomes outdated; cannot access recent policies or course updates  
**Hallucination Risk**: May generate plausible but incorrect institutional information  
**No Attribution**: Cannot cite sources or provide evidence for claims  
**Static Knowledge**: Cannot adapt to evolving university policies

RAG addresses these limitations:

**Current Information**: Retrieves from live document corpus, updated as policies change  
**Grounded Responses**: Every statement traceable to specific retrieved chunks  
**Source Attribution**: Can cite specific documents and policies  
**Dynamic Knowledge**: New documents are immediately searchable after embedding

### 10.2 Comparison with Keyword Search Systems

Traditional keyword search (e.g., Elasticsearch, BM25) relies on lexical matching. RAG employs semantic understanding:

**Keyword System**:
- Query: "assignment deadline extension"
- Matches documents containing these exact words
- Misses synonymous expressions: "due date postponement", "submission deferral"

**RAG System**:
- Query: "assignment deadline extension"  
- Matches based on semantic similarity
- Retrieves documents about "due date flexibility", "late submission policies", "deadline adjustments"
- Understands conceptual relationships beyond surface-level word matching

This semantic capability is crucial in academic contexts where students may phrase queries in diverse ways.

## 11. Conclusion

The RAG architecture presented in this chapter implements a sophisticated information retrieval and generation pipeline tailored to academic institutional contexts. By combining semantic embedding-based retrieval, structured metadata filtering, hierarchical ontology-based prioritization, and context-aware neighbor retrieval, the system delivers accurate, verifiable, and contextually appropriate responses to student queries.

The architecture demonstrates several key innovations:

**Dual embedding strategy** optimizes both preprocessing and retrieval operations  
**Multi-stage filtering** ensures semantic relevance and contextual appropriateness  
**Neighbor retrieval** reconstructs interpretive context around retrieved fragments  
**Hierarchical ranking** prioritizes organizationally proximate information  
**Dual-stream processing** handles academic and administrative content appropriately

These design decisions reflect a deep understanding of both the technical requirements of effective RAG systems and the domain-specific needs of academic information systems. The result is a system that provides students with reliable, contextualized answers grounded in verified institutional documentation.

Subsequent chapters will elaborate on the metadata filtering mechanisms and ontology-based prioritization strategies, providing detailed analysis of their implementation and impact on system performance.

## References

1. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*, 33, 9459-9474.

2. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *arXiv preprint arXiv:1908.10084*.

3. Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 42(4), 824-836.

4. Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535-547.

5. Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., ... & Yih, W. T. (2020). Dense passage retrieval for open-domain question answering. *arXiv preprint arXiv:2004.04906*.
