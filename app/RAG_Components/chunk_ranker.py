from typing import List, Dict, Tuple
from datetime import datetime
from app.models.chat import StudentQueryRequest
import json

class NonAcademicChunkRanker:
    """
    Hierarchical ranking component for non-academic documents
    Prioritizes: Department > Faculty > University
    Secondary factor: Document freshness
    """
    
    # Primary scoring: Hierarchical level importance
    HIERARCHICAL_SCORES = {
        'department': 100,    # Highest - specific to student's department
        'faculty': 50,        # Medium - student's faculty level
        'university': 20,     # Lowest - general university information
        'all': 20,           # Same as university (applies to all)
        'none': 20           # Same as university (no specific level)
    }
    
    # Secondary scoring: Document freshness (max 15% of hierarchical score)
    MAX_FRESHNESS_POINTS = 15
    
    def __init__(self):
        """Initialize the chunk ranker"""
        pass
    
    def rank_chunks(self, filtered_chunks: List[Dict], student_profile: StudentQueryRequest) -> List[Dict]:
        """
        Rank non-academic chunks based on hierarchical level and freshness
        
        Args:
            filtered_chunks: List of chunk dictionaries with metadata
            student_profile: Student information for hierarchical matching
            
        Returns:
            List of chunks sorted by total score (highest first)
        """
        print(f"\n=== NON-ACADEMIC CHUNK RANKING ===")
        print(f"Input chunks: {len(filtered_chunks)}")
        print(f"Student profile: Dept={student_profile.department}, Faculty={student_profile.faculty}")
        
        scored_chunks = []
        
        for chunk_data in filtered_chunks:
            chunk_text = chunk_data.get('text', '')
            chunk_metadata = chunk_data.get('metadata', {})
            
            # Calculate hierarchical score
            hierarchical_score = self._calculate_hierarchical_score(chunk_metadata, student_profile)
            
            # Calculate freshness score  
            freshness_score = self._calculate_freshness_score(chunk_metadata.get('upload_date', ''))
            
            # Total score
            total_score = hierarchical_score + freshness_score
            
            scored_chunk = {
                'text': chunk_text,
                'metadata': chunk_metadata,
                'hierarchical_score': hierarchical_score,
                'freshness_score': freshness_score,
                'total_score': total_score,
                'ranking_details': {
                    'hierarchical_level': self._determine_hierarchical_level(chunk_metadata, student_profile),
                    'document_age_days': self._calculate_age_days(chunk_metadata.get('upload_date', '')),
                    'document_name': chunk_metadata.get('document_name', 'Unknown')
                }
            }
            
            scored_chunks.append(scored_chunk)
            
            # Debug logging
            print(f"Chunk: {chunk_metadata.get('document_name', 'Unknown')[:30]}...")
            print(f"  Hierarchical: {hierarchical_score} ({scored_chunk['ranking_details']['hierarchical_level']})")
            print(f"  Freshness: {freshness_score:.1f} ({scored_chunk['ranking_details']['document_age_days']} days)")
            print(f"  Total Score: {total_score:.1f}")
        
        # Sort by total score (descending)
        ranked_chunks = sorted(scored_chunks, key=lambda x: x['total_score'], reverse=True)
        
        print(f"\n=== RANKING RESULTS ===")
        for i, chunk in enumerate(ranked_chunks[:5], 1):  # Show top 5
            print(f"{i}. {chunk['ranking_details']['document_name'][:40]} (Score: {chunk['total_score']:.1f})")
        
        print(f"Total ranked chunks: {len(ranked_chunks)}\n")
        
        return ranked_chunks
    
    def _calculate_hierarchical_score(self, metadata: Dict, student: StudentQueryRequest) -> int:
        """Calculate hierarchical priority score"""
        
        hierarchical_level = self._determine_hierarchical_level(metadata, student)
        return self.HIERARCHICAL_SCORES.get(hierarchical_level, self.HIERARCHICAL_SCORES['university'])
    
    def _determine_hierarchical_level(self, metadata: Dict, student: StudentQueryRequest) -> str:
        """Determine the hierarchical level of the document"""
        
        # Check departments
        departments_str = metadata.get('departments_affecting', '').strip()
        if departments_str and departments_str.lower() not in ['all', '']:
            try:
                # Parse JSON or comma-separated departments
                if departments_str.startswith('['):
                    departments = json.loads(departments_str)
                else:
                    departments = [d.strip() for d in departments_str.split(',') if d.strip()]
                
                # Check if student's department is specifically mentioned
                if student.department in departments:
                    return 'department'
            except:
                # Fallback to comma-separated parsing
                departments = [d.strip() for d in departments_str.split(',') if d.strip()]
                if student.department in departments:
                    return 'department'
        
        # Check faculties
        faculties_str = metadata.get('faculties_affecting', '').strip()
        if faculties_str and faculties_str.lower() not in ['all', '']:
            try:
                # Parse JSON or comma-separated faculties
                if faculties_str.startswith('['):
                    faculties = json.loads(faculties_str)
                else:
                    faculties = [f.strip() for f in faculties_str.split(',') if f.strip()]
                
                # Check if student's faculty is specifically mentioned
                if student.faculty in faculties:
                    return 'faculty'
            except:
                # Fallback to comma-separated parsing
                faculties = [f.strip() for f in faculties_str.split(',') if f.strip()]
                if student.faculty in faculties:
                    return 'faculty'
        
        # Default to university level
        return 'university'
    
    def _calculate_freshness_score(self, upload_date: str) -> float:
        """Calculate freshness score based on document age"""
        
        if not upload_date:
            return self.MAX_FRESHNESS_POINTS * 0.1  # Very low score for unknown dates
        
        try:
            doc_date = datetime.strptime(upload_date, '%Y-%m-%dT%H:%M:%S.%f')
            current_date = datetime.now()
            age_days = (current_date - doc_date).days
            
            # Freshness scoring tiers
            if age_days <= 30:          # Very fresh: 0-30 days
                return self.MAX_FRESHNESS_POINTS
            elif age_days <= 90:        # Fresh: 31-90 days  
                return self.MAX_FRESHNESS_POINTS * 0.75
            elif age_days <= 180:       # Moderate: 91-180 days
                return self.MAX_FRESHNESS_POINTS * 0.5
            elif age_days <= 365:       # Old: 181-365 days
                return self.MAX_FRESHNESS_POINTS * 0.25
            else:                       # Very old: >365 days
                return self.MAX_FRESHNESS_POINTS * 0.1
                
        except Exception as e:
            print(f"Date parsing error for '{upload_date}': {e}")
            return self.MAX_FRESHNESS_POINTS * 0.1
    
    def _calculate_age_days(self, upload_date: str) -> int:
        """Calculate document age in days for reporting"""
        if not upload_date:
            return 999  # Unknown age
        
        try:
            doc_date = datetime.strptime(upload_date, '%Y-%m-%dT%H:%M:%S.%f')
            current_date = datetime.now()
            return (current_date - doc_date).days
        except:
            return 999
    
    def get_top_chunks(self, ranked_chunks: List[Dict], top_k: int = 5) -> List[str]:
        """
        Extract top-k chunk texts from ranked results
        
        Args:
            ranked_chunks: Output from rank_chunks()
            top_k: Number of top chunks to return
            
        Returns:
            List of chunk texts, highest ranked first
        """
        top_chunks = ranked_chunks[:top_k]
        return [chunk['text'] for chunk in top_chunks]
    
    def get_ranking_summary(self, ranked_chunks: List[Dict]) -> Dict:
        """Generate summary statistics of ranking results"""
        
        if not ranked_chunks:
            return {}
        
        # Count by hierarchical level
        level_counts = {}
        freshness_scores = []
        
        for chunk in ranked_chunks:
            level = chunk['ranking_details']['hierarchical_level']
            level_counts[level] = level_counts.get(level, 0) + 1
            freshness_scores.append(chunk['freshness_score'])
        
        return {
            'total_chunks': len(ranked_chunks),
            'level_distribution': level_counts,
            'average_freshness_score': sum(freshness_scores) / len(freshness_scores),
            'top_score': ranked_chunks[0]['total_score'] if ranked_chunks else 0,
            'score_range': {
                'highest': max(chunk['total_score'] for chunk in ranked_chunks),
                'lowest': min(chunk['total_score'] for chunk in ranked_chunks)
            }
        }