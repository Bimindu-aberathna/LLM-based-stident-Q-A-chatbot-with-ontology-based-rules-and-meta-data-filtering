# Document Corpus Analysis & Metadata Implementation

## Question 1: Document Corpus Search Issue

### Problem
Current implementation creates separate ChromaDB collections per document (`document_name=file.filename`). This creates isolated silos where:
- User A uploads Document1 → stored in "Document1" collection
- User B asks a question → cannot search across all documents, only within specific collections
- Cross-document knowledge retrieval is impossible

### Solution
**Use a single shared collection for all documents** with document identification via metadata.

#### Implementation Changes:

```python
# Instead of:
database = ChromaDB(document_name=file.filename)

# Use:
database = ChromaDB(collection_name="document_corpus")  # Single shared collection
```

#### Metadata Storage:
```python
# Store chunks with document metadata
for i, chunk in enumerate(chunks):
    metadata = {
        "document_name": file.filename,
        "chunk_index": i,
        "upload_timestamp": datetime.now().isoformat(),
        "uploader_id": user_id,  # if available
        # ... other metadata from Question 2
    }
    database.store_vector_with_metadata(embeddings[i], chunk, metadata)
```

#### Query Implementation:
```python
# RAG search across entire corpus
def search_corpus(query: str, filters: dict = None):
    query_embedding = embedder.embed_query(query)
    results = database.similarity_search(
        query_embedding, 
        collection_name="document_corpus",
        filters=filters  # optional filtering
    )
    return results
```

---

## Question 2: Metadata Structure & Filtering

### Proposed Data Models

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime

class AcademicMetadata(BaseModel):
    name: str
    date: str
    type: Literal["Academic"] = "Academic"
    category: str  # "Assignment", "Lecture", "Exam", etc.
    course_code: str
    batches_affecting: List[str]
    degree_programs: List[str]
    specialization_track: Optional[str] = None
    departments_affecting: List[str]
    faculties_affecting: List[str]
    link: str

class NonAcademicMetadata(BaseModel):
    name: str
    date: str
    validity: Optional[str] = None
    type: Literal["Non Academic"] = "Non Academic"
    sub_type: str  # "Time Table", "Notice", "Policy", etc.
    hierarchy_level: str  # "University", "Faculty", "Department", etc.
    batches_affecting: List[str]
    degree_programs: List[str]
    departments_affecting: List[str]
    link: str

class DocumentMetadata(BaseModel):
    # Common fields
    document_id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    upload_timestamp: datetime = Field(default_factory=datetime.now)
    uploader_id: Optional[str] = None
    
    # Specific metadata (union type)
    content_metadata: AcademicMetadata | NonAcademicMetadata
```

### Storage Implementation

```python
# Enhanced document upload with metadata
@router.post("/upload-with-metadata")
async def upload_document_with_metadata(
    file: UploadFile = File(...),
    metadata: DocumentMetadata
):
    # ... existing text processing ...
    
    database = ChromaDB(collection_name="document_corpus")
    
    for i, chunk in enumerate(chunks):
        chunk_metadata = {
            # Document-level metadata
            "document_id": metadata.document_id,
            "filename": metadata.filename,
            "upload_timestamp": metadata.upload_timestamp.isoformat(),
            "uploader_id": metadata.uploader_id,
            
            # Content-specific metadata
            "name": metadata.content_metadata.name,
            "date": metadata.content_metadata.date,
            "type": metadata.content_metadata.type,
            "link": metadata.content_metadata.link,
            
            # Chunk-specific
            "chunk_index": i,
            "chunk_id": f"{metadata.document_id}_chunk_{i}",
        }
        
        # Add type-specific fields
        if isinstance(metadata.content_metadata, AcademicMetadata):
            chunk_metadata.update({
                "category": metadata.content_metadata.category,
                "course_code": metadata.content_metadata.course_code,
                "batches_affecting": metadata.content_metadata.batches_affecting,
                "degree_programs": metadata.content_metadata.degree_programs,
                "specialization_track": metadata.content_metadata.specialization_track,
                "departments_affecting": metadata.content_metadata.departments_affecting,
                "faculties_affecting": metadata.content_metadata.faculties_affecting,
            })
        elif isinstance(metadata.content_metadata, NonAcademicMetadata):
            chunk_metadata.update({
                "validity": metadata.content_metadata.validity,
                "sub_type": metadata.content_metadata.sub_type,
                "hierarchy_level": metadata.content_metadata.hierarchy_level,
                "batches_affecting": metadata.content_metadata.batches_affecting,
                "degree_programs": metadata.content_metadata.degree_programs,
                "departments_affecting": metadata.content_metadata.departments_affecting,
            })
        
        database.store_vector_with_metadata(embeddings[i], chunk, chunk_metadata)
```

### Filtering During Retrieval

```python
def build_filters(user_context: dict) -> dict:
    """Build ChromaDB filters based on user context"""
    filters = {"$and": []}
    
    # Academic filtering
    if user_context.get("type") == "Academic":
        academic_filters = []
        
        if user_context.get("degree_program"):
            academic_filters.append({
                "degree_programs": {"$contains": user_context["degree_program"]}
            })
        
        if user_context.get("department"):
            academic_filters.append({
                "departments_affecting": {"$contains": user_context["department"]}
            })
        
        if user_context.get("batch"):
            academic_filters.append({
                "batches_affecting": {"$contains": user_context["batch"]}
            })
        
        filters["$and"].extend(academic_filters)
    
    # Non-academic filtering
    elif user_context.get("type") == "Non Academic":
        non_academic_filters = []
        
        # Validity check
        if user_context.get("check_validity", True):
            today = datetime.now().isoformat()[:10]  # YYYY-MM-DD
            non_academic_filters.append({
                "$or": [
                    {"validity": {"$eq": None}},
                    {"validity": {"$gte": today}}
                ]
            })
        
        if user_context.get("batch"):
            non_academic_filters.append({
                "batches_affecting": {"$contains": user_context["batch"]}
            })
        
        filters["$and"].extend(non_academic_filters)
    
    return filters if filters["$and"] else {}

# Usage in search
def search_with_context(query: str, user_context: dict):
    filters = build_filters(user_context)
    query_embedding = embedder.embed_query(query)
    
    results = database.similarity_search(
        query_embedding,
        collection_name="document_corpus",
        filters=filters,
        limit=10
    )
    return results

# Example usage:
user_context = {
    "type": "Academic",
    "degree_program": "Bsc(Hons) IT",
    "department": "IM",
    "batch": "2021/22"
}

results = search_with_context("What is lemmatization?", user_context)
```

### Query Examples

```python
# Example 1: Academic query
academic_context = {
    "type": "Academic",
    "degree_program": "Bsc(Hons) IT",
    "department": "IM"
}

# Example 2: Non-academic query with validity check
non_academic_context = {
    "type": "Non Academic",
    "batch": "2025",
    "check_validity": True
}

# Example 3: Broad search (no filters)
broad_context = {}
```

### Database Schema Updates

```python
# Update ChromaDB initialization to support metadata filtering
class ChromaDB:
    def __init__(self, collection_name: str = "document_corpus"):
        self.collection_name = collection_name
        # ... existing initialization ...
    
    def store_vector_with_metadata(self, embedding, text, metadata):
        self.collection.add(
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
            ids=[metadata["chunk_id"]]
        )
    
    def similarity_search(self, query_embedding, filters=None, limit=10):
        return self.collection.query(
            query_embeddings=[query_embedding],
            where=filters,
            n_results=limit
        )
```

## Implementation Notes

1. **Migration Strategy**: Existing documents need to be re-processed with metadata
2. **User Interface**: Create forms for metadata input during upload
3. **Default Values**: Handle missing metadata gracefully
4. **Indexing**: Ensure ChromaDB indexes metadata fields for efficient filtering
5. **Validation**: Add validation for date formats, course codes, etc.
6. **Search API**: Create endpoints that accept user context for filtering

This approach enables:
- Cross-document search across entire corpus
- Fine-grained filtering based on academic/administrative context
- Temporal filtering (validity dates)
- Hierarchical access control (department/faculty/batch level)