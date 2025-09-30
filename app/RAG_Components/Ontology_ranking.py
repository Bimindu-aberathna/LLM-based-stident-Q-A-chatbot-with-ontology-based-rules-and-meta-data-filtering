from typing import List, Dict
from datetime import datetime
import json
from app.models.chat import StudentQueryRequest


def apply_hierarchical_ranking(non_academic_chunks: List[Dict], student: StudentQueryRequest) -> List[Dict]:
    """
    Apply hierarchical ranking to non-academic chunks
    Department > Faculty > University + Freshness scoring
    """
    HIERARCHICAL_SCORES = {
        'department': 100,
        'faculty': 50,
        'university': 20,
        'all': 20
    }
    MAX_FRESHNESS_POINTS = 15
    scored_chunks = []

    for chunk in non_academic_chunks:
        metadata = chunk['metadata']

        # Hierarchy score
        hierarchical_level = determine_hierarchical_level(metadata, student)
        hierarchical_score = HIERARCHICAL_SCORES.get(hierarchical_level, 20)

        # Freshness score
        freshness_score = calculate_freshness_score(metadata.get('upload_date', ''), MAX_FRESHNESS_POINTS)

        # Total score
        total_score = hierarchical_score + freshness_score
        scored_chunks.append({
            'text': chunk['text'],
            'metadata': metadata,
            'similarity_score': chunk['similarity_score'],
            'hierarchical_score': hierarchical_score,
            'freshness_score': freshness_score,
            'total_score': total_score,
            'hierarchical_level': hierarchical_level
        })

    ranked_chunks = sorted(scored_chunks, key=lambda x: x['total_score'], reverse=True)
    return ranked_chunks


def determine_hierarchical_level(metadata: Dict, student: StudentQueryRequest) -> str:
    """Determine hierarchical level: department > faculty > university"""
    departments_str = metadata.get('departments_affecting', '').strip()
    if departments_str and departments_str.lower() not in ['all', '']:
        departments = parse_list_field(departments_str)
        if student.department in departments:
            return 'department'

    faculties_str = metadata.get('faculties_affecting', '').strip()
    if faculties_str and faculties_str.lower() not in ['all', '']:
        faculties = parse_list_field(faculties_str)
        if student.faculty in faculties:
            return 'faculty'

    return 'university'


def calculate_freshness_score(upload_date: str, max_points: int) -> float:
    """Calculate freshness score based on document age"""
    if not upload_date:
        return max_points * 0.1
    try:
        doc_date = datetime.strptime(upload_date, '%Y-%m-%dT%H:%M:%S.%f')
        age_days = (datetime.now() - doc_date).days

        if age_days <= 30:
            return max_points
        elif age_days <= 90:
            return max_points * 0.75
        elif age_days <= 180:
            return max_points * 0.5
        elif age_days <= 365:
            return max_points * 0.25
        else:
            return max_points * 0.1
    except Exception:
        return max_points * 0.1


def parse_list_field(field_value: str) -> List[str]:
    """Helper to parse list fields that might be JSON strings or comma-separated"""
    if not field_value:
        return []
    field_value = field_value.strip()

    if field_value.lower() == 'all':
        return ['all']

    try:
        if field_value.startswith('[') and field_value.endswith(']'):
            parsed = json.loads(field_value)
            return parsed if isinstance(parsed, list) else [str(parsed)]
        else:
            return [item.strip() for item in field_value.split(',') if item.strip()]
    except json.JSONDecodeError:
        return [item.strip() for item in field_value.split(',') if item.strip()]
