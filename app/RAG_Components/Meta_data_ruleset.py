"""
Metadata filtering ruleset for document access control.
Contains all the filtering rules for academic and non-academic documents.
"""

from typing import Dict, List
from app.models.chat import StudentQueryRequest
import json


class MetadataRuleset:
    """
    Class containing all metadata filtering rules for document access control.
    Supports both academic and non-academic document filtering.
    """

    def passes_all_rules(self, doc_metadata: Dict, student: StudentQueryRequest) -> bool:
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
    
    def _parse_list_field(self, field_value: str, field_name: str) -> List[str]:
        """Helper to parse list fields that might be JSON strings or comma-separated"""
        if not field_value:
            return []
        
        field_value = field_value.strip()
        
        # Handle "all" case
        if field_value.lower() == 'all':
            return ['all']
        
        try:
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

    # ==================== LEGACY RULE METHODS (for backward compatibility) ====================
    
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
