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
    def retrieve_similar_with_metadata(self, query_vector: List[float], studentMetadata: StudentQueryRequest, top_k: int = 5, similarity_threshold: float = 0.7) -> List[str]:
        """
        Retrieve similar chunks with rule-based metadata filtering AND similarity threshold
        
        Args:
            similarity_threshold: Minimum cosine similarity score (0.0 to 1.0)
                                0.7 = Good relevance, 0.8 = High relevance, 0.9 = Very high relevance
        """
        try:
            # Debug: Check collection contents
            collection_count = self.collection.count()
            if collection_count == 0:
                return []
            sample_results = self.collection.get(limit=3, include=['documents', 'metadatas'])
            print(f"Sample metadata: {sample_results['metadatas'][:2] if sample_results['metadatas'] else 'None'}")
            
            # Get more results initially to account for filtering
            search_k = min(top_k * 5, 100)  # Increased for better filtering
            
            print(f"Querying with vector length: {len(query_vector)}")
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=search_k,
                include=['documents', 'metadatas', 'distances']  # Include distances for similarity scores
            )
            
            if not results['documents'] or len(results['documents']) == 0:
                print("No documents returned from similarity search")
                return []
            
            print(f"Raw query results: {len(results['documents'][0])} documents found")
            
            # Apply similarity threshold filtering FIRST
            similarity_filtered_results = self._apply_similarity_threshold(results, similarity_threshold)
            print(f"After similarity threshold ({similarity_threshold}): {len(similarity_filtered_results['documents']) if similarity_filtered_results['documents'] else 0} documents remain")
            
            if not similarity_filtered_results['documents'] or len(similarity_filtered_results['documents']) == 0:
                print("No documents passed similarity threshold")
                return []
            
            # Then apply rule-based filtering
            rule_filtered_results = self._apply_rule_based_filters(similarity_filtered_results, studentMetadata)
            print(f"After rule-based filtering: {len(rule_filtered_results)} documents remain")
            
            # Return top_k results after all filtering
            return rule_filtered_results[:top_k]
            
        except Exception as e:
            logging.error(f"Error in retrieve_similar_with_metadata: {str(e)}")
            print(f"Exception in retrieve_similar_with_metadata: {str(e)}")
            return []

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

    def _apply_rule_based_filters(self, results: Dict, studentMetadata: StudentQueryRequest) -> List[str]:
        """
        Apply strict rule-based filtering with hierarchical ranking for non-academic documents
        """
        if not results['documents'] or len(results['documents']) == 0:
            return []
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        distances = results['distances'][0] if results['distances'] else []
        
        # Separate chunks by type during filtering
        academic_chunks = []
        non_academic_chunks = []
        
        for doc, metadata, distance in zip(documents, metadatas, distances):
            if self._passes_all_rules(metadata, studentMetadata):
                similarity_score = 1 - distance  # Convert distance to similarity
                chunk_data = {
                    'text': doc,
                    'metadata': metadata,
                    'similarity_score': similarity_score
                }
                
                if metadata.get('type', '').lower() == 'academic':
                    academic_chunks.append(chunk_data)
                else:
                    non_academic_chunks.append(chunk_data)
        
        print(f"Chunks after rule filtering - Academic: {len(academic_chunks)}, Non-Academic: {len(non_academic_chunks)}")
        
        # Apply hierarchical ranking to non-academic chunks
        if non_academic_chunks:
            ranked_non_academic = self._apply_hierarchical_ranking(non_academic_chunks, studentMetadata)
        else:
            ranked_non_academic = []
        
        # Combine results: Academic chunks (by similarity) + Ranked non-academic chunks
        final_chunks = []
        
        # Add academic chunks (sorted by similarity)
        academic_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
        for chunk in academic_chunks:
            final_chunks.append(chunk['text'])
        
        # Add ranked non-academic chunks
        for chunk in ranked_non_academic:
            chunk['text'] = chunk['text']+ f" RANKING SCORE: {chunk['total_score']:.1f}"
            final_chunks.append(chunk['text'])
            print(final_chunks)
        
        return final_chunks

    def _passes_all_rules(self, doc_metadata: Dict, student: StudentQueryRequest) -> bool:
        """Check if document passes ALL filtering rules based on document type"""
        
        print(f"\n=== FILTERING DOCUMENT ===")
        print(f"Document: {doc_metadata.get('document_name', 'Unknown')}")
        doc_type = doc_metadata.get('type', '').lower()
        print(f"Document Type: {doc_type}")
        
        if doc_type == "academic":
            return self._check_academic_rules(doc_metadata, student)
        elif doc_type == "non-academic":
            return self._check_non_academic_rules(doc_metadata, student)
        else:
            print(f"❌ Document REJECTED: Unknown document type '{doc_type}'")
            return False
    
    def _check_academic_rules(self, doc_metadata: Dict, student: StudentQueryRequest) -> bool:
        """Check all rules for academic documents"""
        print(f"=== ACADEMIC DOCUMENT RULES ===")
        
        # Rule 1: Course code must be in student's course codes
        if not self._check_academic_course_code_rule(doc_metadata, student):
            print(f"❌ Academic Document REJECTED by Course Code Rule")
            return False
        
        # Rule 2: Batch matching
        if not self._check_academic_batch_rule(doc_metadata, student):
            print(f"❌ Academic Document REJECTED by Batch Rule")
            return False
        
        # Rule 3: Year and Semester matching
        if not self._check_academic_year_semester_rule(doc_metadata, student):
            print(f"❌ Academic Document REJECTED by Year/Semester Rule")
            return False
        
        # Rule 4: Degree program matching
        if not self._check_academic_degree_program_rule(doc_metadata, student):
            print(f"❌ Academic Document REJECTED by Degree Program Rule")
            return False
        
        # Rule 5: Specialization track matching
        if not self._check_academic_specialization_rule(doc_metadata, student):
            print(f"❌ Academic Document REJECTED by Specialization Rule")
            return False
        
        # Rule 6: Department matching
        if not self._check_academic_department_rule(doc_metadata, student):
            print(f"❌ Academic Document REJECTED by Department Rule")
            return False
        
        # Rule 7: Faculty matching
        if not self._check_academic_faculty_rule(doc_metadata, student):
            print(f"❌ Academic Document REJECTED by Faculty Rule")
            return False
        
        print(f"✅ Academic Document PASSED all filtering rules")
        return True
    
    def _check_non_academic_rules(self, doc_metadata: Dict, student: StudentQueryRequest) -> bool:
        """Check all rules for non-academic documents"""
        print(f"=== NON-ACADEMIC DOCUMENT RULES ===")
        
        # Rule 1: Validity check
        if not self._check_non_academic_validity_rule(doc_metadata):
            print(f"❌ Non-Academic Document REJECTED by Validity Rule")
            return False
        
        # Rule 2: Batch matching
        if not self._check_non_academic_batch_rule(doc_metadata, student):
            print(f"❌ Non-Academic Document REJECTED by Batch Rule")
            return False
        
        # Rule 3: Specialization track matching
        if not self._check_non_academic_specialization_rule(doc_metadata, student):
            print(f"❌ Non-Academic Document REJECTED by Specialization Rule")
            return False
        
        # Rule 4: Faculty matching
        if not self._check_non_academic_faculty_rule(doc_metadata, student):
            print(f"❌ Non-Academic Document REJECTED by Faculty Rule")
            return False
        
        # Rule 5: Degree program matching
        if not self._check_non_academic_degree_program_rule(doc_metadata, student):
            print(f"❌ Non-Academic Document REJECTED by Degree Program Rule")
            return False
        
        # Rule 6: Department matching
        if not self._check_non_academic_department_rule(doc_metadata, student):
            print(f"❌ Non-Academic Document REJECTED by Department Rule")
            return False
        
        print(f"✅ Non-Academic Document PASSED all filtering rules")
        return True

    # ==================== HELPER METHODS ====================
    
    def _apply_hierarchical_ranking(self, non_academic_chunks: List[Dict], student: StudentQueryRequest) -> List[Dict]:
        """
        Apply hierarchical ranking to non-academic chunks
        Department > Faculty > University + Freshness scoring
        """
        from datetime import datetime
        import json
        
        print(f"\n=== NON-ACADEMIC HIERARCHICAL RANKING ===")
        print(f"Input chunks: {len(non_academic_chunks)}")
        print(f"Student: Dept={student.department}, Faculty={student.faculty}")
        
        # Scoring system
        HIERARCHICAL_SCORES = {
            'department': 100,  # Highest priority
            'faculty': 50,      # Medium priority  
            'university': 20,   # Lowest priority
            'all': 20          # Same as university
        }
        MAX_FRESHNESS_POINTS = 15
        
        scored_chunks = []
        
        for chunk in non_academic_chunks:
            metadata = chunk['metadata']
            
            # Calculate hierarchical level score
            hierarchical_level = self._determine_hierarchical_level(metadata, student)
            hierarchical_score = HIERARCHICAL_SCORES.get(hierarchical_level, 20)
            
            # Calculate freshness score
            freshness_score = self._calculate_freshness_score(metadata.get('upload_date', ''), MAX_FRESHNESS_POINTS)
            
            # Total score
            total_score = hierarchical_score + freshness_score
            
            scored_chunk = {
                'text': chunk['text'],
                'metadata': metadata,
                'similarity_score': chunk['similarity_score'],
                'hierarchical_score': hierarchical_score,
                'freshness_score': freshness_score,
                'total_score': total_score,
                'hierarchical_level': hierarchical_level
            }
            
            scored_chunks.append(scored_chunk)
            
            # Debug logging
            doc_name = metadata.get('document_name', 'Unknown')[:30]
            print(f"  {doc_name}: {hierarchical_level} ({hierarchical_score}) + fresh ({freshness_score:.1f}) = {total_score:.1f}")
        
        # Sort by total score (descending)
        ranked_chunks = sorted(scored_chunks, key=lambda x: x['total_score'], reverse=True)
        
        print(f"Ranking complete: {len(ranked_chunks)} chunks sorted")
        print(f"Top chunk: {ranked_chunks[0]['hierarchical_level']} level (Score: {ranked_chunks[0]['total_score']:.1f})")
        
        return ranked_chunks
    
    def _determine_hierarchical_level(self, metadata: Dict, student: StudentQueryRequest) -> str:
        """Determine hierarchical level: department > faculty > university"""
        
        # Check departments
        departments_str = metadata.get('departments_affecting', '').strip()
        if departments_str and departments_str.lower() not in ['all', '']:
            departments = self._parse_list_field(departments_str, 'departments_affecting')
            if student.department in departments:
                return 'department'
        
        # Check faculties
        faculties_str = metadata.get('faculties_affecting', '').strip()
        if faculties_str and faculties_str.lower() not in ['all', '']:
            faculties = self._parse_list_field(faculties_str, 'faculties_affecting')
            if student.faculty in faculties:
                return 'faculty'
        
        # Default to university level
        return 'university'
    
    def _calculate_freshness_score(self, upload_date: str, max_points: int) -> float:
        """Calculate freshness score based on document age"""
        if not upload_date:
            return max_points * 0.1
        
        try:
            from datetime import datetime
            doc_date = datetime.strptime(upload_date, '%Y-%m-%dT%H:%M:%S.%f')
            age_days = (datetime.now() - doc_date).days
            
            if age_days <= 30:          # Very fresh
                return max_points
            elif age_days <= 90:        # Fresh  
                return max_points * 0.75
            elif age_days <= 180:       # Moderate
                return max_points * 0.5
            elif age_days <= 365:       # Old
                return max_points * 0.25
            else:                       # Very old
                return max_points * 0.1
        except:
            return max_points * 0.1
    
    def _parse_list_field(self, field_value: str, field_name: str) -> List[str]:
        """Helper to parse list fields that might be JSON strings or comma-separated"""
        if not field_value:
            return []
        
        field_value = field_value.strip()
        
        # Handle "all" case
        if field_value.lower() == 'all':
            return ['all']
        
        try:
            import json
            if field_value.startswith('[') and field_value.endswith(']'):
                # It's a JSON array string
                parsed = json.loads(field_value)
                return parsed if isinstance(parsed, list) else [str(parsed)]
            else:
                # It's a comma-separated string
                return [item.strip() for item in field_value.split(',') if item.strip()]
        except json.JSONDecodeError:
            # Fallback to comma-separated parsing
            return [item.strip() for item in field_value.split(',') if item.strip()]
    
    # ==================== ACADEMIC DOCUMENT RULES ====================
    
    def _check_academic_course_code_rule(self, doc_metadata: Dict, student: StudentQueryRequest) -> bool:
        """Academic Rule 1: Course code must be in student's course codes"""
        doc_course_code = doc_metadata.get('course_code', '').strip()
        
        print(f"Academic Course Code Rule Check:")
        print(f"  Document course code: '{doc_course_code}'")
        print(f"  Student course codes: {student.course_codes}")
        
        if not doc_course_code:
            print(f"  Result: PASS (no course code specified)")
            return True
        
        # Normalize course codes (remove spaces for comparison)
        doc_course_normalized = doc_course_code.replace(' ', '').upper()
        normalized_student_courses = [course.replace(' ', '').upper() for course in student.course_codes]
        
        result = doc_course_normalized in normalized_student_courses
        print(f"  Result: {'PASS' if result else 'FAIL'}")
        return result
    
    def _check_academic_batch_rule(self, doc_metadata: Dict, student: StudentQueryRequest) -> bool:
        """Academic Rule 2: Student's batch must be in affecting batches"""
        doc_batches_str = doc_metadata.get('batches_affecting', '')
        
        print(f"Academic Batch Rule Check:")
        print(f"  Document batches: '{doc_batches_str}'")
        print(f"  Student batch: '{student.batch}'")
        
        if not doc_batches_str:
            print(f"  Result: PASS (no batch restrictions)")
            return True
        
        # Use helper to parse JSON or comma-separated
        doc_batches = self._parse_list_field(doc_batches_str, 'batches_affecting')
        print(f"  Parsed batches: {doc_batches}")
        
        result = student.batch in doc_batches
        print(f"  Result: {'PASS' if result else 'FAIL'}")
        return result
    
    def _check_academic_year_semester_rule(self, doc_metadata: Dict, student: StudentQueryRequest) -> bool:
        """Academic Rule 3: Year and semester matching"""
        doc_year = doc_metadata.get('year')
        doc_semester = doc_metadata.get('semester')
        
        print(f"Academic Year/Semester Rule Check:")
        print(f"  Document year: {doc_year}, semester: {doc_semester}")
        print(f"  Student year: {student.current_year}, semester: {student.current_sem}")
        
        if not doc_year:
            print(f"  Result: PASS (no year restriction)")
            return True
        
        try:
            doc_year_int = int(doc_year)
            student_year_int = int(student.current_year)
            
            # FromYear must be less than or equal to student's current year
            if doc_year_int > student_year_int:
                print(f"  Result: FAIL (document year {doc_year_int} > student year {student_year_int})")
                return False
            
            # If FromYear is less than current year, no semester check needed
            if doc_year_int < student_year_int:
                print(f"  Result: PASS (document year {doc_year_int} < student year {student_year_int})")
                return True
            
            # If FromYear == current year, check semester
            if doc_semester and student.current_sem:
                try:
                    doc_sem_int = int(doc_semester)
                    student_sem_int = int(student.current_sem)
                    
                    # FromSemester must be less than or equal to student's current semester
                    result = doc_sem_int <= student_sem_int
                    print(f"  Result: {'PASS' if result else 'FAIL'} (semester check: {doc_sem_int} <= {student_sem_int})")
                    return result
                except (ValueError, TypeError):
                    print(f"  Result: PASS (semester comparison failed, assuming valid)")
                    return True
            else:
                print(f"  Result: PASS (same year, no semester restrictions)")
                return True
                
        except (ValueError, TypeError):
            print(f"  Result: PASS (year comparison failed, assuming valid)")
            return True
    
    def _check_academic_degree_program_rule(self, doc_metadata: Dict, student: StudentQueryRequest) -> bool:
        """Academic Rule 4: Student's degree program must be in affecting degree programs"""
        doc_programs_str = doc_metadata.get('degree_programs', '')
        
        print(f"Academic Degree Program Rule Check:")
        print(f"  Document degree programs: '{doc_programs_str}'")
        print(f"  Student degree program: '{student.degree_program}'")
        
        if not doc_programs_str:
            print(f"  Result: PASS (no degree program restrictions)")
            return True
        
        # Use helper to parse JSON or comma-separated
        doc_programs = self._parse_list_field(doc_programs_str, 'degree_programs')
        print(f"  Parsed programs: {doc_programs}")
        
        result = student.degree_program in doc_programs
        print(f"  Result: {'PASS' if result else 'FAIL'}")
        return result
        return result
    
    def _check_academic_specialization_rule(self, doc_metadata: Dict, student: StudentQueryRequest) -> bool:
        """Academic Rule 5: Specialization track matching (No specialization = applies to all)"""
        doc_specialization = doc_metadata.get('specialization_track', '').strip()
        
        print(f"Academic Specialization Rule Check:")
        print(f"  Document specialization: '{doc_specialization}'")
        print(f"  Student specialization: '{student.specialization}'")
        
        # No specialization track for document means applies to all
        if not doc_specialization:
            print(f"  Result: PASS (no specialization restriction)")
            return True
        
        student_specialization = student.specialization.strip() if student.specialization else ''
        
        result = doc_specialization.lower() == student_specialization.lower()
        print(f"  Result: {'PASS' if result else 'FAIL'}")
        return result
    
    def _check_academic_department_rule(self, doc_metadata: Dict, student: StudentQueryRequest) -> bool:
        """Academic Rule 6: Student's department must be in affecting departments"""
        doc_departments_str = doc_metadata.get('departments_affecting', '')
        
        print(f"Academic Department Rule Check:")
        print(f"  Document departments: '{doc_departments_str}'")
        print(f"  Student department: '{student.department}'")
        
        if not doc_departments_str:
            print(f"  Result: PASS (no department restrictions)")
            return True
        
        # Use helper to parse JSON or comma-separated
        doc_departments = self._parse_list_field(doc_departments_str, 'departments_affecting')
        print(f"  Parsed departments: {doc_departments}")
        
        result = student.department in doc_departments
        print(f"  Result: {'PASS' if result else 'FAIL'}")
        return result
    
    def _check_academic_faculty_rule(self, doc_metadata: Dict, student: StudentQueryRequest) -> bool:
        """Academic Rule 7: Student's faculty must be in affecting faculties"""
        doc_faculties_str = doc_metadata.get('faculties_affecting', '')
        
        print(f"Academic Faculty Rule Check:")
        print(f"  Document faculties: '{doc_faculties_str}'")
        print(f"  Student faculty: '{student.faculty}'")
        
        if not doc_faculties_str:
            print(f"  Result: PASS (no faculty restrictions)")
            return True
        
        # Use helper to parse JSON or comma-separated
        doc_faculties = self._parse_list_field(doc_faculties_str, 'faculties_affecting')
        print(f"  Parsed faculties: {doc_faculties}")
        
        result = student.faculty in doc_faculties
        print(f"  Result: {'PASS' if result else 'FAIL'}")
        return result
    
    # ==================== NON-ACADEMIC DOCUMENT RULES ====================
    
    def _check_non_academic_validity_rule(self, doc_metadata: Dict) -> bool:
        """Non-Academic Rule 1: Validity date check"""
        validity = doc_metadata.get('validity', '').strip()
        
        print(f"Non-Academic Validity Rule Check:")
        print(f"  Document validity: '{validity}'")
        
        if not validity:
            print(f"  Result: PASS (no validity restriction)")
            return True
        
        try:
            from datetime import datetime
            # Parse validity date (format: YYYY-MM-DD)
            validity_date = datetime.strptime(validity, '%Y-%m-%d')
            current_date = datetime.now()
            
            # Document is valid if current date is before or equal to validity date
            result = current_date.date() <= validity_date.date()
            print(f"  Current date: {current_date.date()}")
            print(f"  Validity date: {validity_date.date()}")
            print(f"  Result: {'PASS' if result else 'FAIL'}")
            return result
        except (ValueError, TypeError) as e:
            print(f"  Result: PASS (validity date parsing failed: {e})")
            return True
    
    def _check_non_academic_batch_rule(self, doc_metadata: Dict, student: StudentQueryRequest) -> bool:
        """Non-Academic Rule 2: Batch matching (Empty/'all' = applies to all)"""
        doc_batches_str = doc_metadata.get('batches_affecting', '').strip()
        
        print(f"Non-Academic Batch Rule Check:")
        print(f"  Document batches: '{doc_batches_str}'")
        print(f"  Student batch: '{student.batch}'")
        
        # Empty or 'all' means applies to all
        if not doc_batches_str or doc_batches_str.lower() == 'all':
            print(f"  Result: PASS (applies to all batches)")
            return True
        
        # Use helper to parse JSON or comma-separated
        doc_batches = self._parse_list_field(doc_batches_str, 'batches_affecting')
        print(f"  Parsed batches: {doc_batches}")
        
        result = student.batch in doc_batches
        print(f"  Result: {'PASS' if result else 'FAIL'}")
        return result
    
    def _check_non_academic_specialization_rule(self, doc_metadata: Dict, student: StudentQueryRequest) -> bool:
        """Non-Academic Rule 3: Specialization matching (Empty/'all' = applies to all)"""
        doc_specialization = doc_metadata.get('specialization_track', '').strip()
        
        print(f"Non-Academic Specialization Rule Check:")
        print(f"  Document specialization: '{doc_specialization}'")
        print(f"  Student specialization: '{student.specialization}'")
        
        # Empty or 'all' means applies to all
        if not doc_specialization or doc_specialization.lower() == 'all':
            print(f"  Result: PASS (applies to all specializations)")
            return True
        
        student_specialization = student.specialization.strip() if student.specialization else ''
        
        result = doc_specialization.lower() == student_specialization.lower()
        print(f"  Result: {'PASS' if result else 'FAIL'}")
        return result
    
    def _check_non_academic_faculty_rule(self, doc_metadata: Dict, student: StudentQueryRequest) -> bool:
        """Non-Academic Rule 4: Faculty matching (Empty/'all' = applies to all)"""
        doc_faculties_str = doc_metadata.get('faculties_affecting', '').strip()
        
        print(f"Non-Academic Faculty Rule Check:")
        print(f"  Document faculties: '{doc_faculties_str}'")
        print(f"  Student faculty: '{student.faculty}'")
        
        # Empty or 'all' means applies to all
        if not doc_faculties_str or doc_faculties_str.lower() == 'all':
            print(f"  Result: PASS (applies to all faculties)")
            return True
        
        # Use helper to parse JSON or comma-separated
        doc_faculties = self._parse_list_field(doc_faculties_str, 'faculties_affecting')
        print(f"  Parsed faculties: {doc_faculties}")
        
        result = student.faculty in doc_faculties
        print(f"  Result: {'PASS' if result else 'FAIL'}")
        return result
    
    def _check_non_academic_degree_program_rule(self, doc_metadata: Dict, student: StudentQueryRequest) -> bool:
        """Non-Academic Rule 5: Degree program matching (Empty/'all' = applies to all)"""
        doc_programs_str = doc_metadata.get('degree_programs', '').strip()
        
        print(f"Non-Academic Degree Program Rule Check:")
        print(f"  Document degree programs: '{doc_programs_str}'")
        print(f"  Student degree program: '{student.degree_program}'")
        
        # Empty or 'all' means applies to all
        if not doc_programs_str or doc_programs_str.lower() == 'all':
            print(f"  Result: PASS (applies to all degree programs)")
            return True
        
        # Parse JSON array or comma-separated programs
        doc_programs = self._parse_list_field(doc_programs_str, 'degree_programs')
        print(f"  Parsed degree programs: {doc_programs}")
        
        result = student.degree_program in doc_programs
        print(f"  Result: {'PASS' if result else 'FAIL'}")
        return result
    
    def _check_non_academic_department_rule(self, doc_metadata: Dict, student: StudentQueryRequest) -> bool:
        """Non-Academic Rule 6: Department matching (Empty/'all' = applies to all)"""
        doc_departments_str = doc_metadata.get('departments_affecting', '').strip()
        
        print(f"Non-Academic Department Rule Check:")
        print(f"  Document departments: '{doc_departments_str}'")
        print(f"  Student department: '{student.department}'")
        
        # Empty or 'all' means applies to all
        if not doc_departments_str or doc_departments_str.lower() == 'all':
            print(f"  Result: PASS (applies to all departments)")
            return True
        
        # Parse JSON array or comma-separated departments
        doc_departments = self._parse_list_field(doc_departments_str, 'departments_affecting')
        print(f"  Parsed departments: {doc_departments}")
        
        result = student.department in doc_departments
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
            #if not a number,
            if not doc_year.isdigit():
                return True
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

    def _check_validity_rule(self, doc_metadata: Dict):
        """Rule 9: Validity check for non-academic documents"""
        validity = doc_metadata.get('validity')
        if not validity:
            print(f"🚨🚨🚨🚨  No validity restriction")
            return True  # No validity restriction
        
        try:
            from datetime import datetime
            # Parse validity date (assume ISO format)
            validity_date = datetime.fromisoformat(validity.replace('Z', '+00:00'))
            current_date = datetime.now()
            print(f"🚨🚨🚨🚨Validity Rule Check: Current date={current_date}, Validity date={validity_date}")
            print(f"🚨🚨🚨🚨Validity Rule Check: {current_date <= validity_date}")
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