"""
Education matching logic.
"""

from difflib import get_close_matches
import re

from app.engine.text_utils import clean_text

DEGREE_LEVELS = [
    {"level": 0, "label": "high_school", "aliases": ["high school", "secondary", "higher secondary", "12th", "10th", "hsc", "ssc"]},
    {"level": 1, "label": "diploma", "aliases": ["diploma", "polytechnic", "associate", "associates", "associate degree", "pg diploma", "post graduate diploma"]},
    {"level": 2, "label": "bachelors", "aliases": ["bachelor", "bachelors", "bachelors degree", "b tech", "btech", "b e", "be", "b sc", "bsc", "b com", "bcom", "b a", "ba", "bca", "bba", "undergraduate", "ug"]},
    {"level": 3, "label": "masters", "aliases": ["master", "masters", "masters degree", "mba", "m tech", "mtech", "m e", "me", "m sc", "msc", "m com", "mcom", "m a", "ma", "mca", "postgraduate", "pg", "ms"]},
    {"level": 4, "label": "doctorate", "aliases": ["phd", "ph d", "doctorate", "doctoral", "dphil"]},
]

ALIAS_TO_LEVEL = {
    alias: degree_level["level"]
    for degree_level in DEGREE_LEVELS
    for alias in degree_level["aliases"]
}

ALL_ALIASES = list(ALIAS_TO_LEVEL.keys())


def normalize_degree(degree: str) -> str:
    """Lowercase and normalize punctuation/spacing for a degree string."""
    cleaned = clean_text(degree).lower()
    cleaned = re.sub(r"[^a-z0-9 ]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def get_degree_level(degree: str) -> int:
    """Return the hierarchy level of a degree."""
    normalized = normalize_degree(degree)
    if not normalized:
        return -1

    if normalized in ALIAS_TO_LEVEL:
        return ALIAS_TO_LEVEL[normalized]

    for alias, level in ALIAS_TO_LEVEL.items():
        if len(alias) > 3 and re.search(rf"\b{re.escape(alias)}\b", normalized):
            return level

    fuzzy_match = get_close_matches(normalized, ALL_ALIASES, n=1, cutoff=0.88)
    if fuzzy_match:
        return ALIAS_TO_LEVEL[fuzzy_match[0]]

    return -1


def get_highest_education(resume_education: list) -> str:
    """Find the highest degree from the candidate's education list."""
    if not resume_education:
        return ""

    fallback = ""
    highest = ""
    highest_level = -1

    for entry in resume_education:
        degree_type = clean_text(entry.get("degree_type", ""))
        degree = clean_text(entry.get("degree", ""))
        field = clean_text(entry.get("field", ""))

        candidate_values = [value for value in [degree_type, degree, field] if value]
        if candidate_values and not fallback:
            fallback = candidate_values[0]

        for candidate_value in candidate_values:
            level = get_degree_level(candidate_value)
            if level > highest_level:
                highest_level = level
                highest = candidate_value

    return highest or fallback


def compute_education_score(resume_education: list, required_education: str) -> dict:
    """Compute education match score."""
    if not required_education:
        return {
            "score": 100,
            "candidate_degree": get_highest_education(resume_education),
            "required_degree": required_education,
            "note": "No education requirement specified",
        }

    candidate_degree = get_highest_education(resume_education)
    candidate_level = get_degree_level(candidate_degree)
    required_level = get_degree_level(required_education)

    if candidate_level == -1 or required_level == -1:
        return {
            "score": 50,
            "candidate_degree": candidate_degree,
            "required_degree": required_education,
            "note": "Could not determine degree level - manual review recommended",
        }

    if candidate_level >= required_level:
        score = 100.0
        note = "Meets or exceeds required education"
    else:
        gap = required_level - candidate_level
        score = max(round(100 - (gap * 20), 2), 0)
        note = f"Below required level by {gap} step(s)"

    return {
        "score": score,
        "candidate_degree": candidate_degree,
        "required_degree": required_education,
        "note": note,
    }
