from typing import List, Dict, Optional
from datetime import datetime, timezone
import json
from app.models.chat import StudentQueryRequest


def apply_hierarchical_ranking(non_academic_chunks: List[Dict], student: StudentQueryRequest) -> List[Dict]:
    """
    Apply hierarchical ranking to non-academic chunks
    Department > Faculty > University + Freshness scoring
    """
    print(f"\n🤢🤢🤢 HIERARCHICAL RANKING 🤢🤢🤢")
    HIERARCHICAL_SCORES = {
        'department': 100,
        'faculty': 50,
        'university': 20,
        'all': 20
    }
    MAX_FRESHNESS_POINTS = 15
    scored_chunks = []

    student_department = (getattr(student, "department", "") or "").strip().lower()
    student_faculty = (getattr(student, "faculty", "") or "").strip().lower()

    for chunk in non_academic_chunks:
        metadata = chunk.get('metadata', {}) or {}
        text = chunk.get('text', '')
        sim = chunk.get('similarity_score', 0.0)

        # Hierarchy score
        hierarchical_level = determine_hierarchical_level(metadata, student, student_department, student_faculty)
        hierarchical_score = HIERARCHICAL_SCORES.get(hierarchical_level, 20)

        # Freshness score (robust parsing)
        upload_raw = _get_upload_date_str(metadata)
        freshness_score, upload_ts = calculate_freshness_score(upload_raw, MAX_FRESHNESS_POINTS, return_timestamp=True)

        total_score = hierarchical_score + freshness_score

        scored_chunks.append({
            'text': text,
            'metadata': metadata,
            'similarity_score': sim,
            'hierarchical_score': hierarchical_score,
            'freshness_score': freshness_score,
            'total_score': total_score,
            'hierarchical_level': hierarchical_level,
            'upload_timestamp': upload_ts
        })

    ranked_chunks = sorted(scored_chunks, key=lambda x: (x['total_score'], x['upload_timestamp']), reverse=True)
    return ranked_chunks

# def determine_hierarchical_level(metadata: Dict, student: StudentQueryRequest,
#                                  student_department: Optional[str] = None,
#                                  student_faculty: Optional[str] = None) -> str:
#     """Determine hierarchical level: department > faculty > university/all"""
#     student_department = (student_department or (getattr(student, "department", "") or "")).strip().lower()
#     student_faculty = (student_faculty or (getattr(student, "faculty", "") or "")).strip().lower()

#     departments_str = (metadata.get('departments_affecting', '') or '').strip()
#     if departments_str and departments_str.lower() not in ['all', '']:
#         departments = parse_list_field(departments_str, to_lower=True)
#         if student_department and student_department in departments:
#             return 'department'

#     faculties_str = (metadata.get('faculties_affecting', '') or '').strip()
#     if faculties_str and faculties_str.lower() not in ['all', '']:
#         faculties = parse_list_field(faculties_str, to_lower=True)
#         if student_faculty and student_faculty in faculties:
#             return 'faculty'

#     scope = (metadata.get('scope', '') or '').strip().lower()
#     if scope == 'all':
#         return 'all'

#     return 'university'

def determine_hierarchical_level(metadata: Dict, student: StudentQueryRequest,
                                 student_department: Optional[str] = None,
                                 student_faculty: Optional[str] = None) -> str:
    """
    Determine level from explicit metadata:
      - department (aliases: dept)
      - faculty   (aliases: fac)
      - university (aliases: uni, institution, campus)
      - all -> treated as university
    Fallback: university
    """
    raw = (
        (metadata or {}).get('hierarchy_level')
        or metadata.get('hierarchy')
        or metadata.get('level')
        or metadata.get('scope')  # sometimes used
        or ""
    )
    raw = str(raw).strip().lower()

    if raw in ("department", "dept"):
        print("\n👻👻👻Detected department level from metadata.")
        return "department"
    if raw in ("faculty", "fac"):
        print("\n👻👻👻Detected faculty level from metadata.")
        return "faculty"
    if raw in ("all", "university", "uni", "institution", "campus"):
        print("\n👻👻👻Detected university level from metadata.")
        return "university"
    print("\n👻👻👻Fallback to university level.")
    return "university"

def calculate_freshness_score(upload_date: str, max_points: int, return_timestamp: bool = False):
    """
    Calculate freshness score based on document age with robust ISO parsing.
    - 0 days -> max_points
    - 365+ days -> ~10% of max_points
    Returns score, and optionally a sortable timestamp (int).
    """
    dt = _parse_iso_datetime(upload_date)
    if not dt:
        score = round(max_points * 0.1, 2)
        return (score, 0) if return_timestamp else score

    now = datetime.now(timezone.utc)
    # Make both timezone-aware in UTC
    if not dt.tzinfo:
        dt = dt.replace(tzinfo=timezone.utc)
    age_days = (now - dt).days
    age_hours = (now - dt).total_seconds() / 3600

    # Base score by age range (piecewise linear)
    if age_days <= 30:
        base = max_points
        bonus = max(0.0, (720 - age_hours) / 720.0 * 2.0)  # up to +2 within 30 days
        score = min(max_points, base + bonus)
    elif age_days <= 90:
        base = max_points * 0.75
        bonus = max(0.0, (90 - age_days) / 90.0 * 1.5)
        score = min(max_points * 0.75 + 1.5, base + bonus)
    elif age_days <= 180:
        base = max_points * 0.5
        bonus = max(0.0, (180 - age_days) / 180.0 * 1.0)
        score = min(max_points * 0.5 + 1.0, base + bonus)
    elif age_days <= 365:
        base = max_points * 0.25
        bonus = max(0.0, (365 - age_days) / 365.0 * 0.5)
        score = min(max_points * 0.25 + 0.5, base + bonus)
    else:
        score = max_points * 0.1

    score = round(score, 2)
    ts = int(dt.timestamp())
    return (score, ts) if return_timestamp else score

def parse_list_field(field_value: str, to_lower: bool = False) -> List[str]:
    """Parse JSON-list or comma-separated values. Optionally lower-case."""
    if not field_value:
        return []
    s = field_value.strip()
    if s.lower() == 'all':
        return ['all']
    # Try JSON list
    try:
        if s.startswith('[') and s.endswith(']'):
            data = json.loads(s)
            items = [str(x).strip() for x in (data if isinstance(data, list) else [data])]
        else:
            items = [item.strip() for item in s.split(',') if item.strip()]
    except json.JSONDecodeError:
        items = [item.strip() for item in s.split(',') if item.strip()]
    return [i.lower() for i in items] if to_lower else items

def _get_upload_date_str(metadata: Dict) -> str:
    """
    Return the best available upload date string from common metadata keys.
    """
    for k in ('upload_date', 'uploaded_at', 'created_at', 'last_modified', 'date', 'timestamp'):
        v = metadata.get(k)
        if v:
            return str(v)
    return ""

def _parse_iso_datetime(val: str) -> Optional[datetime]:
    """
    Robust ISO parser:
    - Supports 'Z' suffix, timezone offsets, missing microseconds
    - Supports date-only 'YYYY-MM-DD'
    - Supports epoch seconds (int/float string)
    """
    if not val:
        return None
    s = str(val).strip()
    # Epoch?
    try:
        if s.isdigit() or (s.replace('.', '', 1).isdigit() and s.count('.') < 2):
            return datetime.fromtimestamp(float(s), tz=timezone.utc)
    except Exception:
        pass
    # Normalize Z
    if s.endswith('Z'):
        s = s[:-1] + '+00:00'
    # Try fromisoformat first
    try:
        return datetime.fromisoformat(s)
    except Exception:
        pass
    # Try common fallback patterns
    patterns = [
        '%Y-%m-%dT%H:%M:%S.%f%z',
        '%Y-%m-%dT%H:%M:%S%z',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d'
    ]
    for p in patterns:
        try:
            dt = datetime.strptime(s, p)
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        except Exception:
            continue
    return None
