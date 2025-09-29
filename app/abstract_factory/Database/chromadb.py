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
    def retrieve_similar_with_metadata(self, query_vector: List[float], studentMetadata: StudentQueryRequest, top_k: int = 5, similarity_threshold: float = 0.7) -> tuple[List[str], List[str]]:
        """
        Retrieve similar chunks with rule-based metadata filtering AND similarity threshold
        Returns: (academic_chunks, non_academic_chunks) - Two separate lists
        """
        try:
            # Debug: Check collection contents
            collection_count = self.collection.count()
            if collection_count == 0:
                return [], []
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
                return [], []
            
            print(f"Raw query results: {len(results['documents'][0])} documents found")
            
            # Apply similarity threshold filtering FIRST
            similarity_filtered_results = self._apply_similarity_threshold(results, similarity_threshold)
            print(f"After similarity threshold ({similarity_threshold}): {len(similarity_filtered_results['documents']) if similarity_filtered_results['documents'] else 0} documents remain")
            
            if not similarity_filtered_results['documents'] or len(similarity_filtered_results['documents']) == 0:
                print("No documents passed similarity threshold")
                return [], []
            
            # Then apply rule-based filtering
            academic_chunks, non_academic_chunks = self._apply_rule_based_filters(similarity_filtered_results, studentMetadata)
            
            print(f"After rule-based filtering: Academic: {len(academic_chunks)}, Non-Academic: {len(non_academic_chunks)}")
            
            # Return both lists (truncated to top_k each)
            return academic_chunks[:10], non_academic_chunks[:10]
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

    def _apply_rule_based_filters(self, results: Dict, studentMetadata: StudentQueryRequest) -> tuple[List[str], List[str]]:
        """
        Apply strict rule-based filtering with hierarchical ranking for non-academic documents
        Returns: (academic_chunks_list, non_academic_chunks_list)
        """
        if not results['documents'] or len(results['documents']) == 0:
            return [], []
        
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
        
        # Process academic chunks with neighbors (sorted by similarity)
        final_academic_list = []
        academic_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
        for chunk in academic_chunks:
            print(f"\n--- Processing Academic Chunk ---")
            print(f"Original chunk preview: {chunk['text'][:100]}...")
            
            chunk_with_neighbors = self.retrieve_neighbor_chunks_for_a_chunk(
                chunk=chunk['text'],
                chunk_type="academic",
                neighbor_count=2  # Only 2 neighbors on each side
            )
            
            if chunk_with_neighbors:
                print(f"Found {len(chunk_with_neighbors)} chunks (including neighbors)")
                for neighbor_chunk in chunk_with_neighbors:
                    final_academic_list.append(neighbor_chunk['text'])
            else:
                print(f"No neighbors found for academic chunk, adding original")
                final_academic_list.append(chunk['text'])
        
        # Process non-academic chunks with neighbors (already ranked)
        final_non_academic_list = []
        for chunk in ranked_non_academic:
            print(f"\n--- Processing Non-Academic Chunk ---")
            print(f"Original chunk preview: {chunk['text'][:100]}...")
            print(f"Chunk score: {chunk['total_score']}")
            
            chunk_with_neighbors = self.retrieve_neighbor_chunks_for_a_chunk(
                chunk=chunk['text'],
                chunk_type="non-academic",
                chunk_score=chunk['total_score'],
                neighbor_count=2  # Only 2 neighbors on each side
            )
            
            if chunk_with_neighbors:
                print(f"Found {len(chunk_with_neighbors)} chunks (including neighbors)")
                for neighbor_chunk in chunk_with_neighbors:
                    if neighbor_chunk['is_original_chunk']:
                        annotated_chunk = f"{neighbor_chunk['text']} [ORIGINAL-SCORE: {chunk['total_score']:.1f}]"
                    else:
                        annotated_chunk = f"{neighbor_chunk['text']} [NEIGHBOR-SCORE: {chunk['total_score']*0.9:.1f}]"
                    final_non_academic_list.append(annotated_chunk)
            else:
                print(f"No neighbors found for non-academic chunk, adding original")
                annotated_chunk = f"{chunk['text']} [SCORE: {chunk['total_score']:.1f}]"
                final_non_academic_list.append(annotated_chunk)
        
        print(f"\n=== BEFORE DEDUPLICATION ===")
        print(f"Academic chunks: {len(final_academic_list)}")
        print(f"Non-academic chunks: {len(final_non_academic_list)}")
        
        # ==================== DEDUPLICATION ====================
        
        # Deduplicate academic chunks while preserving order
        seen_academic = set()
        deduplicated_academic = []
        for chunk in final_academic_list:
            # Create a shorter key for comparison (first 100 characters)
            chunk_key = chunk[:100].strip()
            if chunk_key not in seen_academic:
                seen_academic.add(chunk_key)
                deduplicated_academic.append(chunk)
            else:
                print(f"Removed duplicate academic chunk: {chunk[:50]}...")
        
        # Deduplicate non-academic chunks while preserving order and scores
        seen_non_academic = set()
        deduplicated_non_academic = []
        for chunk in final_non_academic_list:
            # Extract text without score annotation for comparison
            if '[SCORE:' in chunk or '[ORIGINAL-SCORE:' in chunk or '[NEIGHBOR-SCORE:' in chunk:
                # Find the last occurrence of '[' to split text from score
                last_bracket = chunk.rfind('[')
                chunk_text = chunk[:last_bracket].strip() if last_bracket > 0 else chunk
            else:
                chunk_text = chunk
            
            # Create a shorter key for comparison
            chunk_key = chunk_text[:100].strip()
            if chunk_key not in seen_non_academic:
                seen_non_academic.add(chunk_key)
                deduplicated_non_academic.append(chunk)
            else:
                print(f"Removed duplicate non-academic chunk: {chunk[:50]}...")
        
        print(f"\n=== AFTER DEDUPLICATION ===")
        print(f"Academic chunks: {len(deduplicated_academic)} (removed {len(final_academic_list) - len(deduplicated_academic)} duplicates)")
        print(f"Non-academic chunks: {len(deduplicated_non_academic)} (removed {len(final_non_academic_list) - len(deduplicated_non_academic)} duplicates)")
        
        # Show final chunk previews with lengths
        for i, chunk in enumerate(deduplicated_academic[:3], 1):  # Show first 3
            print(f"Academic {i} (len:{len(chunk)}): {chunk[:100]}...")
            
        for i, chunk in enumerate(deduplicated_non_academic[:3], 1):  # Show first 3
            print(f"Non-Academic {i} (len:{len(chunk)}): {chunk[:100]}...")
        
        # Calculate total token estimate (rough: 4 chars = 1 token)
        total_academic_tokens = sum(len(chunk) // 4 for chunk in deduplicated_academic)
        total_non_academic_tokens = sum(len(chunk) // 4 for chunk in deduplicated_non_academic)
        total_tokens = total_academic_tokens + total_non_academic_tokens
        
        print(f"\n=== TOKEN USAGE ESTIMATE ===")
        print(f"Academic tokens: ~{total_academic_tokens}")
        print(f"Non-academic tokens: ~{total_non_academic_tokens}")
        print(f"Total estimated tokens: ~{total_tokens}")
        
        if total_tokens > 6000:
            print(f"⚠️  WARNING: Estimated tokens ({total_tokens}) may exceed context window!")
        
        print(f"Final results - Academic: {len(deduplicated_academic)} chunks, Non-Academic: {len(deduplicated_non_academic)} chunks")
        
        return deduplicated_academic, deduplicated_non_academic

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
        
    def retrieve_neighbor_chunks_for_a_chunk(self, chunk: str, chunk_type: str = 'academic', chunk_score: float = 0, neighbor_count: int = 2) -> List[Dict]:
        """
        Retrieve neighboring chunks for a single given chunk to provide better context
        
        Args:
            chunk: Single chunk text to find neighbors for
            chunk_type: 'academic' or 'non-academic' 
            chunk_score: Score for non-academic chunks (ignored for academic)
            neighbor_count: Number of neighbors to retrieve on each side of target chunk
        
        Returns:
            List of chunk dictionaries with text, metadata, and scores in document order
            Includes the original chunk + its neighbors
        """
        try:
            if not chunk:
                return []
            
            print(f"\n=== NEIGHBOR CHUNK RETRIEVAL ===")
            print(f"Processing single {chunk_type} chunk")
            print(f"Neighbor count: {neighbor_count} on each side")
            print(f"Passed chunk (length: {len(chunk)}):")
            print(f"'{chunk[:200]}...' " + ("(truncated)" if len(chunk) > 200 else ""))
            
            # Get all documents from collection with metadata
            all_results = self.collection.get(
                include=['documents', 'metadatas']
            )
            
            if not all_results['documents']:
                return []
            
            all_documents = all_results['documents']
            all_metadatas = all_results['metadatas']
            
            # Find the target chunk in the collection
            target_metadata = None
            target_index = None
            doc_name = None
            
            for doc, metadata in zip(all_documents, all_metadatas):
                if doc == chunk:  # Found the target chunk
                    target_metadata = metadata
                    target_index = metadata.get('chunk_index', 0)
                    doc_name = metadata.get('document_name', 'Unknown')
                    break
            
            if target_metadata is None:
                print(f"❌ Target chunk not found in collection")
                print(f"Collection has {len(all_documents)} documents")
                print(f"Target chunk length: {len(chunk)}")
                print(f"Searching for exact matches...")
                
                # Try partial matching as fallback
                for i, (doc, metadata) in enumerate(zip(all_documents[:5], all_metadatas[:5])):
                    print(f"  Doc {i}: {len(doc)} chars, starts with: '{doc[:50]}...'")
                    if chunk[:100] in doc or doc[:100] in chunk:
                        print(f"  ✓ Found partial match with doc {i}")
                        target_metadata = metadata
                        target_index = metadata.get('chunk_index', 0)
                        doc_name = metadata.get('document_name', 'Unknown')
                        break
                
                if target_metadata is None:
                    print(f"❌ No matches found - returning empty list")
                    return []
            
            print(f"Found target chunk in document: {doc_name}, index: {target_index}")
            
            # Group all chunks from the same document by chunk_index
            document_chunks = {}
            for doc, metadata in zip(all_documents, all_metadatas):
                if metadata.get('document_name') == doc_name:
                    chunk_index = metadata.get('chunk_index', 0)
                    document_chunks[chunk_index] = {
                        'text': doc,
                        'metadata': metadata,
                        'chunk_index': chunk_index
                    }
            
            # Calculate neighbor range
            max_index = max(document_chunks.keys()) if document_chunks else 0
            start_index = max(0, target_index - neighbor_count)
            end_index = min(max_index, target_index + neighbor_count)
            
            print(f"Retrieving chunks from index {start_index} to {end_index} (max: {max_index})")
            
            # Collect chunks in document order
            result_chunks = []
            for chunk_idx in range(start_index, end_index + 1):
                if chunk_idx in document_chunks:
                    chunk_data = document_chunks[chunk_idx]
                    chunk_text = chunk_data['text']
                    
                    # Create result chunk with appropriate scoring
                    result_chunk = {
                        'text': chunk_text,
                        'metadata': chunk_data['metadata'],
                        'chunk_index': chunk_idx,
                        'document_name': doc_name,
                        'is_original_chunk': chunk_text == chunk  # True only for the target chunk
                    }
                    
                    # Add scoring based on chunk type
                    if chunk_type.lower() == 'non-academic':
                        if chunk_text == chunk:
                            # Original chunk gets the provided score
                            result_chunk['total_score'] = chunk_score
                            result_chunk['hierarchical_score'] = max(0, chunk_score - 15)  # Assume freshness ~15
                            result_chunk['freshness_score'] = min(15, chunk_score)
                        else:
                            # Neighbors get slightly reduced score
                            result_chunk['total_score'] = chunk_score * 0.9
                            result_chunk['hierarchical_score'] = max(0, chunk_score - 15) * 0.9
                            result_chunk['freshness_score'] = min(15, chunk_score) * 0.9
                    else:
                        # Academic chunks don't have hierarchical scores
                        result_chunk['similarity_based'] = True
                    
                    result_chunks.append(result_chunk)
            
            print(f"Retrieved {len(result_chunks)} chunks (including neighbors)")
            print(f"Chunk indices: {[chunk['chunk_index'] for chunk in result_chunks]}")
            
            print(f"\n=== NEIGHBOR CHUNKS RETRIEVED ===")
            for i, result_chunk in enumerate(result_chunks):
                chunk_preview = result_chunk['text'][:100] + ("..." if len(result_chunk['text']) > 100 else "")
                is_original = "ORIGINAL" if result_chunk['is_original_chunk'] else "NEIGHBOR"
                print(f"  {i+1}. [{is_original}] Index {result_chunk['chunk_index']}: '{chunk_preview}'")
            
            return result_chunks
            
        except Exception as e:
            print(f"Error in retrieve_neighbor_chunks_for_a_chunk: {e}")
            import traceback
            traceback.print_exc()
            return []