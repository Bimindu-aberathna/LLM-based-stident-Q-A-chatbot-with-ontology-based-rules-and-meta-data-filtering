from app.abstract_factory.Database.vector_store import VectorStore
from app.RAG_Components.metadata_filtering_manager import apply_rule_based_filters
from app.models.chat import StudentQueryRequest
import chromadb
from typing import List, Optional,Dict, Any
import logging
import uuid
import os

class ChromaDB(VectorStore):
    COLLECTION_NAME = "document_corpus_test"  # Single shared collection for all documents

    def __init__(self):
        self.client = None  # Instance-level client
        self.collection = None
        super().__init__()
        self.chroma_connect()

    def store_vectors(self, vectors: List[List[float]], chunks: List[str], 
                     metadatas: Optional[List[dict]] = None) -> None:
        if len(vectors) != len(chunks):
            raise ValueError("Number of vectors must match number of chunks")
        
        # Generate unique IDs
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        
        # Batch add all documents at once
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=vectors,
            metadatas=metadatas
        )

  
    def chroma_connect(self):
        try:
            
            db_path = "./data/chroma_db"
            os.makedirs(db_path, exist_ok=True)
            
            # Use persistent client
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Connected to persistent ChromaDB at: {db_path}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to ChromaDB: {e}")
        
    """
    class StudentQueryRequest(BaseModel):
    message: str
    batch: str
    department: str
    degree_program: str
    faculty: str
    current_year: str
    current_sem: str
    specialization: str
    course_codes: list[str] = []
    courses_done: list[str] = []
    """
    def retrieve_similar_with_metadata(self, query_vector: List[float], studentMetadata: StudentQueryRequest, top_k: int = 10, similarity_threshold: float = 0.7) -> tuple[List[str], List[str]]:
        """
        Retrieve similar chunks with rule-based metadata filtering AND similarity threshold
        Returns: (academic_chunks, non_academic_chunks) - Two separate lists
        """
        try:
            # Debug: Check collection contents
            collection_count = self.collection.count()
            if collection_count == 0:
                return [], []
            
            # Get more results initially to account for filtering
            search_k = min(top_k * 8, 100)  # Increased for better filtering
            
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=search_k,
                include=['documents', 'metadatas', 'distances']  # Include distances for similarity scores
            )
            
            if not results['documents'] or len(results['documents']) == 0:
                print("No documents returned from similarity search")
                return [], []
            
            print(f"Raw query results: {len(results['documents'][0])} documents found")
            
            # Apply similarity threshold filtering FIRST
            similarity_filtered_results = self._apply_similarity_threshold(results, similarity_threshold)
            
            if not similarity_filtered_results['documents'] or len(similarity_filtered_results['documents']) == 0:
                print("No documents passed similarity threshold")
                return [], []
            
            # Then apply rule-based filtering
            academic_chunks, non_academic_chunks = apply_rule_based_filters(similarity_filtered_results, studentMetadata)
            
            print(f"After rule-based filtering: Academic: {len(academic_chunks)}, Non-Academic: {len(non_academic_chunks)}")
            
            # Return both lists (truncated to top_k each)
            return academic_chunks[:top_k], non_academic_chunks[:top_k]
        except Exception as e:
            logging.error(f"Error in retrieve_similar_with_metadata: {str(e)}")
            print(f"Exception in retrieve_similar_with_metadata: {str(e)}")
            return [], []

    def _apply_similarity_threshold(self, results: Dict, threshold: float) -> Dict:
        """
        Filter results by similarity threshold
        ChromaDB returns distances, need to convert to similarity scores
        """
        if not results['documents'] or len(results['documents']) == 0:
            return {'documents': [], 'metadatas': [], 'distances': []}
        
        filtered_documents = []
        filtered_metadatas = []
        filtered_distances = []
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        distances = results['distances'][0] if results['distances'] else []
        
        print(f"\n=== SIMILARITY THRESHOLD FILTERING ===")
        print(f"Threshold: {threshold}")
        
        for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
            # Convert distance to similarity score
            # ChromaDB with cosine distance: similarity = 1 - distance
            similarity_score = 1.0 - distance
            
            print(f"Document {i+1}: similarity = {similarity_score:.3f} ({'PASS' if similarity_score >= threshold else 'REJECT'})")
            
            if similarity_score >= threshold:
                filtered_documents.append(doc)
                filtered_metadatas.append(metadata)
                filtered_distances.append(distance)
        
        return {
            'documents': [filtered_documents] if filtered_documents else [],
            'metadatas': [filtered_metadatas] if filtered_metadatas else [],
            'distances': [filtered_distances] if filtered_distances else []
        }


    def baseline_retriever(self, query_vector: List[float], studentMetadata: StudentQueryRequest, top_k: int = 10, similarity_threshold: float = 0.7) -> List[str]:
        """
        Retrieve similar chunks without rule-based metadata filtering BUT with similarity threshold for the baseline system
        Returns: List of document chunks
        """
        try:
            # Debug: Check collection contents
            collection_count = self.collection.count()
            if collection_count == 0:
                return []
            
            # Get more results initially to account for filtering
            search_k = min(top_k * 8, 100)  # Increased for better filtering
            
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=search_k,
                include=['documents', 'distances']  # Include distances for similarity scores
            )
            
            if not results['documents'] or len(results['documents']) == 0:
                print("No documents returned from similarity search")
                return []
            
            # Apply similarity threshold filtering
            base_similarity_filtered_results = self._baseline_apply_similarity_threshold(results, similarity_threshold)

            if not base_similarity_filtered_results['documents'] or len(base_similarity_filtered_results['documents']) == 0:
                print("No documents passed similarity threshold")
                return []
            
            # Get all the documents as a single string list
            all_chunks = base_similarity_filtered_results['documents'][0]

            # Return top_k chunks
            return all_chunks[:top_k]
        except Exception as e:
            logging.error(f"Error in baseline_retriever: {str(e)}")
            print(f"Exception in baseline_retriever: {str(e)}")
            return []
        
    def _baseline_apply_similarity_threshold(self, results: Dict, threshold: float) -> Dict:
        """
        Filter results by similarity threshold
        ChromaDB returns distances, need to convert to similarity scores
        """
        if not results['documents'] or len(results['documents']) == 0:
            return {'documents': [], 'distances': []}
        
        filtered_documents = []
        filtered_distances = []
        
        documents = results['documents'][0]
        distances = results['distances'][0] if results['distances'] else []
        
        print(f"\n=== BASELINE SIMILARITY THRESHOLD FILTERING ===")
        print(f"Threshold: {threshold}")
        
        for i, (doc, distance) in enumerate(zip(documents, distances)):
            # Convert distance to similarity score
            # ChromaDB with cosine distance: similarity = 1 - distance
            similarity_score = 1.0 - distance
            
            print(f"Document {i+1}: similarity = {similarity_score:.3f} ({'PASS' if similarity_score >= threshold else 'REJECT'})")
            
            if similarity_score >= threshold:
                filtered_documents.append(doc)
                filtered_distances.append(distance)
        
        return {
            'documents': [filtered_documents] if filtered_documents else [],
            'distances': [filtered_distances] if filtered_distances else []
        }
    

        
    def clear_collection(self) -> None:
        try:
            self.client.delete_collection(name=self.COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logging.error(f"Error clearing collection: {str(e)}")
            raise RuntimeError(f"Failed to clear collection: {e}")
    
    def debug_collection_info(self) -> Dict:
        """Debug method to check collection status"""
        try:
            count = self.collection.count()
            sample_data = self.collection.get(limit=5, include=['documents', 'metadatas'])
            return {
                "count": count,
                "sample_documents": sample_data['documents'][:2] if sample_data['documents'] else [],
                "sample_metadata": sample_data['metadatas'][:2] if sample_data['metadatas'] else []
            }
        except Exception as e:
            return {"error": str(e)}

    def debug_chromadb(self) -> str:
        db = ChromaDB()
        debug_info = []

        debug_info.append("\n=== CHROMADB DEBUG ANALYSIS ===")
        
        # Check total documents
        info = db.debug_collection_info()
        debug_info.append(f"Total documents in collection: {info.get('count', 'Error')}")
        if 'error' in info:
            debug_info.append(f"Error retrieving collection info: {info['error']}")
            return "\n".join(debug_info)
        # Look for documents containing "initially planned"
        docs_with_initially = db.find_documents_by_keyword("initially planned")
        debug_info.append(f"Found {len(docs_with_initially)} documents with 'initially planned'")
        for doc in docs_with_initially:
            debug_info.append(f"Doc {doc['index']+1}:")
            debug_info.append(f"  Upload date: {doc['metadata'].get('upload_date', 'No date')}")
            debug_info.append(f"  Document name: {doc['metadata'].get('document_name', 'No name')}")
            debug_info.append(f"  Preview: {doc['preview']}")
            debug_info.append("")     
            print()
        
        # Look for documents containing "reach agreement" or "reached agreement"
        debug_info.append("\n=== DOCUMENTS CONTAINING 'agreement' ===")
        docs_with_agreement = db.find_documents_by_keyword("agreement")
        debug_info.append(f"Found {len(docs_with_agreement)} documents with 'agreement'")
        for doc in docs_with_agreement:
            debug_info.append(f"Doc {doc['index']+1}:")
            debug_info.append(f"  Upload date: {doc['metadata'].get('upload_date', 'No date')}")
            debug_info.append(f"  Document name: {doc['metadata'].get('document_name', 'No name')}")
            debug_info.append(f"  Preview: {doc['preview']}")
            debug_info.append("")

        # Look for documents containing "futa strike"
        debug_info.append("\n=== DOCUMENTS CONTAINING 'futa' ===")
        docs_with_futa = db.find_documents_by_keyword("futa")
        debug_info.append(f"Found {len(docs_with_futa)} documents with 'futa'")
        for doc in docs_with_futa:
            debug_info.append(f"Doc {doc['index']+1}:")
            debug_info.append(f"  Upload date: {doc['metadata'].get('upload_date', 'No date')}")
            debug_info.append(f"  Document name: {doc['metadata'].get('document_name', 'No name')}")
            debug_info.append(f"  Preview: {doc['preview']}")
            debug_info.append("")
        return "\n".join(debug_info)
    
    def find_documents_by_keyword(self, keyword: str) -> List[Dict]:
        """Debug method to find documents containing specific keywords"""
        try:
            all_docs = self.collection.get(include=['documents', 'metadatas'])
            matching_docs = []
            
            for i, (doc, metadata) in enumerate(zip(all_docs['documents'], all_docs['metadatas'])):
                if keyword.lower() in doc.lower():
                    matching_docs.append({
                        'index': i,
                        'document': doc,
                        'metadata': metadata,
                        'preview': doc[:200] + '...' if len(doc) > 200 else doc
                    })
            
            return matching_docs
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
