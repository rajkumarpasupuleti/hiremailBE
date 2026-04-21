"""
Final score aggregator.
"""

from app.engine.skill_matcher import compute_skill_score
from app.engine.experience_matcher import compute_experience_score
from app.engine.education_matcher import compute_education_score
from app.engine.job_normalizer import normalize_job_payload
from app.engine.keyword_matcher import compute_keyword_score
from app.engine.resume_normalizer import normalize_resume_payload

DEFAULT_WEIGHTS = {
    "skills": 0.45,
    "experience": 0.20,
    "education": 0.10,
    "keywords": 0.25,
}


def validate_weights(weights: dict) -> None:
    """Raise ValueError if weights don't sum to 1.0."""
    total = round(sum(weights.values()), 5)
    if total != 1.0:
        raise ValueError(f"Weights must sum to 1.0, got {total}")


def compute_final_score(resume: dict, job: dict, weights: dict = None) -> dict:
    """Compute the overall ATS match score between a resume and a job."""
    weights = weights or DEFAULT_WEIGHTS
    validate_weights(weights)
    normalized_resume = normalize_resume_payload(resume)
    normalized_job = normalize_job_payload(job)

    skill_result = compute_skill_score(
        resume_skills=normalized_resume.get("skills", []),
        required_skills=normalized_job.get("required_skills", []),
        preferred_skills=normalized_job.get("preferred_skills", []),
    )

    experience_result = compute_experience_score(
        resume_experience=normalized_resume.get("experience", []),
        required_years=normalized_job.get("experience_years_required", 0),
    )

    education_result = compute_education_score(
        resume_education=normalized_resume.get("education", []),
        required_education=normalized_job.get("education_required", ""),
    )

    keyword_result = compute_keyword_score(
        resume=normalized_resume,
        job=normalized_job,
    )

    final_score = round(
        (skill_result["score"] * weights["skills"])
        + (experience_result["score"] * weights["experience"])
        + (education_result["score"] * weights["education"])
        + (keyword_result["score"] * weights["keywords"]),
        2,
    )

    return {
        "final_score": final_score,
        "grade": get_grade(final_score),
        "breakdown": {
            "skills": {
                "weight": f"{int(weights['skills'] * 100)}%",
                "score": skill_result["score"],
                "matched_required": skill_result["matched_required"],
                "missing_required": skill_result["missing_required"],
                "matched_preferred": skill_result["matched_preferred"],
            },
            "experience": {
                "weight": f"{int(weights['experience'] * 100)}%",
                "score": experience_result["score"],
                "candidate_years": experience_result["candidate_years"],
                "required_years": experience_result["required_years"],
                "note": experience_result["note"],
            },
            "education": {
                "weight": f"{int(weights['education'] * 100)}%",
                "score": education_result["score"],
                "candidate_degree": education_result["candidate_degree"],
                "required_degree": education_result["required_degree"],
                "note": education_result["note"],
            },
            "keywords": {
                "weight": f"{int(weights['keywords'] * 100)}%",
                "score": keyword_result["score"],
                "tfidf_similarity": keyword_result["tfidf_similarity"],
                "matched_keywords": keyword_result["matched_keywords"],
                "missing_keywords": keyword_result["missing_keywords"],
            },
        },
    }


def get_grade(score: float) -> str:
    """Convert numeric score to a human-readable grade."""
    if score >= 85:
        return "Excellent Match"
    if score >= 70:
        return "Good Match"
    if score >= 50:
        return "Partial Match"
    return "Poor Match"
