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
        """
        Check if document passes ALL filtering rules
        Returns True only if ALL applicable rules pass
        """
        
        # Rule 1: Batch matching
        if not self._check_batch_rule(doc_metadata, student):
            return False
        
        # Rule 2: Faculty matching
        if not self._check_faculty_rule(doc_metadata, student):
            return False
        
        # Rule 3: Department matching
        if not self._check_department_rule(doc_metadata, student):
            return False
        
        # Rule 4: Degree program matching
        if not self._check_degree_program_rule(doc_metadata, student):
            return False
        
        # Rule 5: Specialization track matching
        if not self._check_specialization_rule(doc_metadata, student):
            return False
        
        # Rule 6: Year matching
        if not self._check_year_rule(doc_metadata, student):
            return False
        
        # Document type-specific rules
        doc_type = doc_metadata.get('type', '').lower()
        
        if doc_type == 'academic':
            # Rule 7: Course relevance for academic documents
            if not self._check_course_relevance_rule(doc_metadata, student):
                return False
            
            # Rule 8: Semester matching for academic documents
            if not self._check_semester_rule(doc_metadata, student):
                return False
        
        elif doc_type == 'non-academic':
            # Rule 9: Validity check for non-academic documents
            if not self._check_validity_rule(doc_metadata):
                return False
        
        return True

    def _check_batch_rule(self, doc_metadata: Dict, student: StudentQueryRequest) -> bool:
        """Rule 1: Batch matching"""
        doc_batches = doc_metadata.get('batches_affecting', '')
        if not doc_batches:
            return True  # No batch restriction
        
        # Split comma-separated string to list
        batch_list = [b.strip() for b in doc_batches.split(',') if b.strip()]
        if not batch_list:
            return True  # Empty list means applicable to all
        
        return student.batch in batch_list

    def _check_faculty_rule(self, doc_metadata: Dict, student: StudentQueryRequest) -> bool:
        """Rule 2: Faculty matching - Handle JSON strings"""
        doc_faculties = doc_metadata.get('faculties_affecting', '')
        if not doc_faculties:
            return True
        
        # Handle JSON string format: ["Science"] -> Science
        try:
            import json
            if doc_faculties.startswith('["') and doc_faculties.endswith('"]'):
                faculty_list = json.loads(doc_faculties)
            else:
                faculty_list = [f.strip() for f in doc_faculties.split(',') if f.strip()]
        except:
            faculty_list = [f.strip() for f in doc_faculties.split(',') if f.strip()]
        
        if not faculty_list:
            return True
        
        print(f"Faculty check: Student={student.faculty}, Doc={faculty_list}")
        return student.faculty in faculty_list

    def _check_department_rule(self, doc_metadata: Dict, student: StudentQueryRequest) -> bool:
        """Rule 3: Department matching - Handle JSON strings"""
        doc_departments = doc_metadata.get('departments_affecting', '')
        if not doc_departments:
            return True
        
        # Handle JSON string format: ["IM"] -> IM
        try:
            import json
            if doc_departments.startswith('["') and doc_departments.endswith('"]'):
                dept_list = json.loads(doc_departments)
            else:
                dept_list = [d.strip() for d in doc_departments.split(',') if d.strip()]
        except:
            dept_list = [d.strip() for d in doc_departments.split(',') if d.strip()]
        
        if not dept_list:
            return True
        
        print(f"Department check: Student={student.department}, Doc={dept_list}")
        return student.department in dept_list

    def _check_degree_program_rule(self, doc_metadata: Dict, student: StudentQueryRequest) -> bool:
        """Rule 4: Degree program matching - Handle JSON strings"""
        doc_programs = doc_metadata.get('degree_programs', '')
        if not doc_programs:
            return True
        
        # Handle JSON string format: ["Bsc(Hons) IT","Bsc(Hons) MIT"] -> list
        try:
            import json
            if doc_programs.startswith('["') and doc_programs.endswith('"]'):
                program_list = json.loads(doc_programs)
            else:
                program_list = [p.strip() for p in doc_programs.split(',') if p.strip()]
        except:
            program_list = [p.strip() for p in doc_programs.split(',') if p.strip()]
        
        if not program_list:
            return True
        
        print(f"Degree program check: Student={student.degree_program}, Doc={program_list}")
        return student.degree_program in program_list

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