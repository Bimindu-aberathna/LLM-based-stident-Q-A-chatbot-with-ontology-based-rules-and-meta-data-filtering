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
        print(f"🤢🤢🤢🤢🤢Chunk text:::: {chunk['text']}🤢🤢🤢🤢")

        # Hierarchy score
        hierarchical_level = determine_hierarchical_level(metadata, student)
        hierarchical_score = HIERARCHICAL_SCORES.get(hierarchical_level, 20)

        # Freshness score
        freshness_score = calculate_freshness_score(metadata.get('upload_date', ''), MAX_FRESHNESS_POINTS)
        
        # Parse upload timestamp for secondary sorting
        upload_timestamp = 0
        try:
            if metadata.get('upload_date'):
                upload_timestamp = datetime.strptime(metadata['upload_date'], '%Y-%m-%dT%H:%M:%S.%f').timestamp()
        except Exception:
            upload_timestamp = 0
        
        # Total score
        total_score = hierarchical_score + freshness_score
        scored_chunks.append({
            'text': chunk['text'],
            'metadata': metadata,
            'similarity_score': chunk['similarity_score'],
            'hierarchical_score': hierarchical_score,
            'freshness_score': freshness_score,
            'total_score': total_score,
            'hierarchical_level': hierarchical_level,
            'upload_timestamp': upload_timestamp  # ✅ ADD THIS LINE - IT WAS MISSING!
        })

    ranked_chunks = sorted(scored_chunks, key=lambda x: (x['total_score'], x['upload_timestamp']), reverse=True)
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
    """Calculate freshness score based on document age with granular bonus for newer docs"""
    if not upload_date:
        return max_points * 0.1
    try:
        doc_date = datetime.strptime(upload_date, '%Y-%m-%dT%H:%M:%S.%f')
        age_days = (datetime.now() - doc_date).days
        age_hours = (datetime.now() - doc_date).total_seconds() / 3600

        # Base score by age range
        if age_days <= 30:
            base_score = max_points
        elif age_days <= 90:
            base_score = max_points * 0.75
        elif age_days <= 180:
            base_score = max_points * 0.5
        elif age_days <= 365:
            base_score = max_points * 0.25
        else:
            base_score = max_points * 0.1

        # Granular bonus within each range (newer = slightly higher score)
        if age_days <= 30:
            # Within 30 days: bonus based on hours (newer gets up to +2 points)
            hour_bonus = max(0, (720 - age_hours) / 720 * 2)  # 720 hours = 30 days
            return min(max_points, base_score + hour_bonus)
        elif age_days <= 90:
            # Within 90 days: bonus based on days (newer gets up to +1.5 points)
            day_bonus = max(0, (90 - age_days) / 90 * 1.5)
            return min(max_points * 0.75 + 1.5, base_score + day_bonus)
        elif age_days <= 180:
            # Within 180 days: bonus based on days (newer gets up to +1 point)
            day_bonus = max(0, (180 - age_days) / 180 * 1.0)
            return min(max_points * 0.5 + 1.0, base_score + day_bonus)
        elif age_days <= 365:
            # Within 365 days: bonus based on days (newer gets up to +0.5 points)
            day_bonus = max(0, (365 - age_days) / 365 * 0.5)
            return min(max_points * 0.25 + 0.5, base_score + day_bonus)
        else:
            # Older than 1 year: minimal bonus
            return base_score

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
