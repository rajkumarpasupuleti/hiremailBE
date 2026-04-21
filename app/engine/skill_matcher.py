"""
Skill matching logic.

Compares candidate skills from parsed resume against
required and preferred skills from the job description.
"""

import re

from app.engine.text_utils import clean_text


def normalize(skill: str) -> str:
    """Lowercase, remove special chars, and strip a skill string for comparison."""
    cleaned = clean_text(skill).lower()
    cleaned = re.sub(r"[^a-z0-9 ]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def compute_skill_score(resume_skills: list, required_skills: list, preferred_skills: list = None) -> dict:
    """
    Compute skill match score.
    """
    if not required_skills:
        return {"score": 0, "matched_required": [], "matched_preferred": [], "missing_required": []}

    resume_set = set(normalize(s) for s in resume_skills)
    required_set = set(normalize(s) for s in required_skills)
    preferred_set = set(normalize(s) for s in (preferred_skills or []))

    matched_required = required_set & resume_set
    missing_required = required_set - resume_set
    matched_preferred = preferred_set & resume_set

    required_score = len(matched_required) / len(required_set) * 100

    preferred_bonus = 0
    if preferred_set:
        preferred_bonus = (len(matched_preferred) / len(preferred_set)) * 10

    final_score = min(round(required_score + preferred_bonus, 2), 100)

    return {
        "score": final_score,
        "matched_required": sorted(matched_required),
        "missing_required": sorted(missing_required),
        "matched_preferred": sorted(matched_preferred),
        "required_match_pct": round(required_score, 2),
        "preferred_bonus": round(preferred_bonus, 2),
    }
