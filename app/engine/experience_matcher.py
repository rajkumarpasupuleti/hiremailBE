"""
Experience matching logic.
"""


def get_total_experience_years(resume_experience: list) -> float:
    """Sum up total years of experience from all experience entries."""
    total = 0.0
    for entry in resume_experience:
        years = entry.get("years", 0)
        try:
            total += float(years)
        except (TypeError, ValueError):
            continue
    return total


def compute_experience_score(resume_experience: list, required_years: int) -> dict:
    """Compute experience match score."""
    if not required_years or required_years <= 0:
        return {
            "score": 100,
            "candidate_years": get_total_experience_years(resume_experience),
            "required_years": required_years,
            "note": "No experience requirement specified",
        }

    candidate_years = get_total_experience_years(resume_experience)

    if candidate_years >= required_years:
        score = 100.0
        note = "Meets or exceeds required experience"
    else:
        score = round((candidate_years / required_years) * 100, 2)
        note = f"Has {candidate_years} yr(s), needs {required_years} yr(s)"

    return {
        "score": score,
        "candidate_years": candidate_years,
        "required_years": required_years,
        "note": note,
    }
