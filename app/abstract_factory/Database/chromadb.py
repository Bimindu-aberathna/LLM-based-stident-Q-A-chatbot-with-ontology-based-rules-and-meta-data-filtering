from app.abstract_factory.Database.vector_store import VectorStore
from app.models.chat import StudentQueryRequest
import chromadb
from typing import List, Optional,Dict, Any
import logging
import uuid
import os

class ChromaDB(VectorStore):
    COLLECTION_NAME = "document_corpus"  # Single shared collection for all documents

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

    def retrieve_similar(self, query_vector: List[float], studentMetadata: StudentQueryRequest, top_k: int = 5) -> List[str]:
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k
            )
            # Return the actual text chunks, not embeddings
            return results['documents'][0] if results['documents'] else []
        except Exception as e:
            print(f"Error retrieving similar chunks: {e}")
            return []
    
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
    def retrieve_similar_with_metadata(self, query_vector: List[float], studentMetadata: StudentQueryRequest, top_k: int = 5) -> List[str]:
        """
        Retrieve similar chunks with rule-based metadata filtering
        """
        try:
            # Debug: Check collection contents
            collection_count = self.collection.count()
            print(f"Collection '_________________________{self.COLLECTION_NAME}' contains {collection_count} documents")
            
            if collection_count == 0:
                print("WARNING:_________________________ Collection is empty! No documents to search.")
                return []
            
            # Debug: Show some sample documents
            sample_results = self.collection.get(limit=3, include=['documents', 'metadatas'])
            # print(f"Sample documents in collection: {sample_results['documents'][:2] if sample_results['documents'] else 'None'}")
            print(f"Sample metadata: {sample_results['metadatas'][:2] if sample_results['metadatas'] else 'None'}")
            
            # Get more results initially to account for filtering
            search_k = min(top_k * 3, 50)  # Get 3x more results to filter from
            
            print(f"Querying with vector length: {len(query_vector)}")
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=search_k
            )
            print(f"Raw query results: _________________________{len(results.get('documents', [[]])[0])} documents found")
            
            if not results['documents'] or len(results['documents']) == 0:
                print("No documents returned from similarity search")
                return []
            
            print(f"Retrieved following documents: {results['documents'][0][:2] if results['documents'][0] else 'None'}")  # Show first 2 only
            print(f"With corresponding metadatas: {results['metadatas'][0][:2] if results['metadatas'] and results['metadatas'][0] else 'None'}")
            
            # Apply rule-based filtering
            filtered_results = self._apply_rule_based_filters(results, studentMetadata)
            print(f"After filtering: {len(filtered_results)} documents remain")
            
            # Return top_k results after filtering
            return filtered_results[:top_k]
            
        except Exception as e:
            logging.error(f"Error in retrieve_similar_with_metadata: {str(e)}")
            print(f"Exception in retrieve_similar_with_metadata: {str(e)}")
            return []

    def _apply_rule_based_filters(self, results: Dict, studentMetadata: StudentQueryRequest) -> List[str]:
        """
        Apply strict rule-based filtering according to the defined ruleset
        """
        if not results['documents'] or len(results['documents']) == 0:
            return []
        
        filtered_documents = []
        documents = results['documents'][0]
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        
        for doc, metadata in zip(documents, metadatas):
            if self._passes_all_rules(metadata, studentMetadata):
                filtered_documents.append(doc)
        
        return filtered_documents

    def _passes_all_rules(self, doc_metadata: Dict, student: StudentQueryRequest) -> bool:
        """Check if document passes ALL filtering rules with detailed logging"""
        
        print(f"\n=== FILTERING DOCUMENT ===")
        print(f"Document: {doc_metadata.get('document_name', 'Unknown')}")
        print(f"Document Type: {doc_metadata.get('type', 'Unknown')}")
        
        # Rule 1: Batch matching
        if not self._check_batch_rule(doc_metadata, student):
            print(f"❌ Document REJECTED by Batch Rule")
            return False
        
        # Rule 2: Faculty matching  
        if not self._check_faculty_rule(doc_metadata, student):
            print(f"❌ Document REJECTED by Faculty Rule")
            return False
        
        # Rule 3: Department matching
        if not self._check_department_rule(doc_metadata, student):
            print(f"❌ Document REJECTED by Department Rule")
            return False
        
        # Rule 4: Degree program matching
        if not self._check_degree_program_rule(doc_metadata, student):
            print(f"❌ Document REJECTED by Degree Program Rule")
            return False
        
        print(f"✅ Document PASSED all filtering rules")
        return True

    def _check_faculty_rule(self, doc_metadata: Dict, student: StudentQueryRequest) -> bool:
        """Rule 2: Faculty matching - Handle 'all' case"""
        doc_faculties_str = doc_metadata.get('faculties_affecting', '')
        
        print(f"Faculty Rule Check:")
        print(f"  Document faculties: '{doc_faculties_str}'")
        print(f"  Student faculty: '{student.faculty}'")
        
        # If document applies to all faculties
        if doc_faculties_str == "all" or not doc_faculties_str:
            print(f"  Result: PASS (document applies to all faculties)")
            return True
        
        # Parse comma-separated faculties
        doc_faculties = [f.strip() for f in doc_faculties_str.split(',') if f.strip()]
        
        if not doc_faculties:
            print(f"  Result: PASS (no faculty restrictions)")
            return True
        
        result = student.faculty in doc_faculties
        print(f"  Result: {'PASS' if result else 'FAIL'}")
        return result

    def _check_department_rule(self, doc_metadata: Dict, student: StudentQueryRequest) -> bool:
        """Rule 3: Department matching - Handle 'all' case"""
        doc_departments_str = doc_metadata.get('departments_affecting', '')
        
        print(f"Department Rule Check:")
        print(f"  Document departments: '{doc_departments_str}'")
        print(f"  Student department: '{student.department}'")
        
        # If document applies to all departments
        if doc_departments_str == "all" or not doc_departments_str:
            print(f"  Result: PASS (document applies to all departments)")
            return True
        
        # Parse comma-separated departments
        doc_departments = [d.strip() for d in doc_departments_str.split(',') if d.strip()]
        
        if not doc_departments:
            print(f"  Result: PASS (no department restrictions)")
            return True
        
        result = student.department in doc_departments
        print(f"  Result: {'PASS' if result else 'FAIL'}")
        return result

    def _check_degree_program_rule(self, doc_metadata: Dict, student: StudentQueryRequest) -> bool:
        """Rule 4: Degree program matching - Handle 'all' case"""
        doc_programs_str = doc_metadata.get('degree_programs', '')
        
        print(f"Degree Program Rule Check:")
        print(f"  Document degree programs: '{doc_programs_str}'")
        print(f"  Student degree program: '{student.degree_program}'")
        
        # If document applies to all degree programs
        if doc_programs_str == "all" or not doc_programs_str:
            print(f"  Result: PASS (document applies to all degree programs)")
            return True
        
        # Parse comma-separated programs
        doc_programs = [p.strip() for p in doc_programs_str.split(',') if p.strip()]
        
        if not doc_programs:
            print(f"  Result: PASS (no degree program restrictions)")
            return True
        
        result = student.degree_program in doc_programs
        print(f"  Result: {'PASS' if result else 'FAIL'}")
        return result

    def _check_batch_rule(self, doc_metadata: Dict, student: StudentQueryRequest) -> bool:
        """Rule 1: Batch matching - Handle 'all' case"""
        doc_batches_str = doc_metadata.get('batches_affecting', '')
        
        print(f"Batch Rule Check:")
        print(f"  Document batches: '{doc_batches_str}'")
        print(f"  Student batch: '{student.batch}'")
        
        # If document applies to all batches
        if doc_batches_str == "all" or not doc_batches_str:
            print(f"  Result: PASS (document applies to all batches)")
            return True
        
        # Parse comma-separated batches
        doc_batches = [b.strip() for b in doc_batches_str.split(',') if b.strip()]
        
        if not doc_batches:
            print(f"  Result: PASS (no batch restrictions)")
            return True
        
        result = student.batch in doc_batches
        print(f"  Result: {'PASS' if result else 'FAIL'}")
        return result

    def _check_course_relevance_rule(self, doc_metadata: Dict, student: StudentQueryRequest) -> bool:
        """Rule 7: Course relevance - Handle spaces in course codes"""
        doc_course_code = doc_metadata.get('course_code', '').strip()
        if not doc_course_code:
            return True
        
        # Normalize course codes (remove spaces for comparison)
        doc_course_normalized = doc_course_code.replace(' ', '')
        
        # Check against student's current and completed courses (also normalized)
        all_student_courses = student.course_codes + student.courses_done
        normalized_student_courses = [course.replace(' ', '') for course in all_student_courses]
        
        print(f"Course check: Student courses={normalized_student_courses}, Doc course={doc_course_normalized}")
        return doc_course_normalized in normalized_student_courses

    def _check_year_rule(self, doc_metadata: Dict, student: StudentQueryRequest) -> bool:
        """Rule 6: Academic year matching - More flexible"""
        doc_year = doc_metadata.get('year')
        if not doc_year:
            return True
        
        try:
            doc_year_int = int(doc_year)
            student_year_int = int(student.current_year)
            
            if(student_year_int < doc_year_int):
                return False
            else :
                return True
        except (ValueError, TypeError):
            return True

    def _check_specialization_rule(self, doc_metadata: Dict, student: StudentQueryRequest) -> bool:
        """Rule 5: Specialization track matching"""
        doc_specialization = doc_metadata.get('specialization_track', '').strip()
        if not doc_specialization:
            return True  # No specialization restriction
        
        student_specialization = student.specialization.strip() if student.specialization else ''
        if not student_specialization:
            return False  # Student has no specialization but document requires one
        
        return doc_specialization.lower() == student_specialization.lower()

    def _check_semester_rule(self, doc_metadata: Dict, student: StudentQueryRequest) -> bool:
        """Rule 8: Semester matching for academic documents"""
        doc_semester = doc_metadata.get('semester')
        if not doc_semester:
            return True  # No semester restriction
        
        try:
            doc_sem_int = int(doc_semester)
            student_sem_int = int(student.current_sem)
            
            # Document semester should be current semester or below
            return doc_sem_int <= student_sem_int
        except (ValueError, TypeError):
            return True  # If conversion fails, don't filter out

    def _check_validity_rule(self, doc_metadata: Dict) -> bool:
        """Rule 9: Validity check for non-academic documents"""
        validity = doc_metadata.get('validity')
        if not validity:
            return True  # No validity restriction
        
        try:
            from datetime import datetime
            # Parse validity date (assume ISO format)
            validity_date = datetime.fromisoformat(validity.replace('Z', '+00:00'))
            current_date = datetime.now()
            
            # Document is valid if current date is before validity date
            return current_date <= validity_date
        except (ValueError, TypeError):
            return True  # If parsing fails, assume valid


        
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