"""
Keyword and text similarity matching.
"""

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.engine.text_utils import clean_text


def build_text_from_resume(resume: dict) -> str:
    """Flatten a parsed resume dict into a single text string for TF-IDF."""
    parts = []
    parts.extend(resume.get("skills", []))

    summary = resume.get("summary", "")
    if summary:
        parts.append(summary)

    for exp in resume.get("experience", []):
        desc = exp.get("description", "")
        title = exp.get("title", "")
        if desc:
            parts.append(desc)
        if title:
            parts.append(title)

    for edu in resume.get("education", []):
        degree = edu.get("degree", "")
        field = edu.get("field", "")
        if degree:
            parts.append(degree)
        if field:
            parts.append(field)

    parts.extend(resume.get("certifications", []))

    for project in resume.get("projects", []):
        name = project.get("name", "")
        desc = project.get("description", "")
        if name:
            parts.append(name)
        if desc:
            parts.append(desc)

    return clean_text(" ".join(str(part) for part in parts if part))


def build_text_from_job(job: dict) -> str:
    """Flatten a job dict into a single text string for TF-IDF."""
    parts = []
    parts.append(job.get("title", ""))
    parts.append(job.get("description", ""))
    parts.extend(job.get("required_skills", []))
    parts.extend(job.get("preferred_skills", []))
    parts.extend(job.get("keywords", []))
    return clean_text(" ".join(str(part) for part in parts if part))


def compute_tfidf_similarity(resume_text: str, job_text: str) -> float:
    """Compute cosine similarity between resume and job texts using TF-IDF."""
    resume_text = clean_text(resume_text)
    job_text = clean_text(job_text)

    if not resume_text or not job_text:
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return round(float(similarity[0][0]), 4)
    except ValueError:
        return 0.0


def keyword_in_resume(kw: str, resume_words: set, resume_lower: str) -> bool:
    """Check if a keyword exists in the resume."""
    kw_lower = clean_text(kw).lower()
    if " " in kw_lower:
        return kw_lower in resume_lower
    return kw_lower in resume_words


def tokenize(text: str) -> set:
    """Tokenize text into lowercase alphanumeric words."""
    return set(re.findall(r"[a-z0-9]+", clean_text(text).lower()))


def compute_keyword_overlap(resume_text: str, job_keywords: list) -> dict:
    """Check how many of the job's explicit keywords appear in the resume text."""
    if not job_keywords:
        return {"score": 100, "matched": [], "missing": []}

    resume_lower = clean_text(resume_text).lower()
    resume_words = tokenize(resume_lower)

    matched = [kw for kw in job_keywords if keyword_in_resume(kw, resume_words, resume_lower)]
    missing = [kw for kw in job_keywords if not keyword_in_resume(kw, resume_words, resume_lower)]
    score = round(len(matched) / len(job_keywords) * 100, 2)

    return {
        "score": score,
        "matched": matched,
        "missing": missing,
    }


def compute_keyword_score(resume: dict, job: dict) -> dict:
    """Compute keyword/text similarity score."""
    resume_text = build_text_from_resume(resume)
    job_text = build_text_from_job(job)

    tfidf_score = compute_tfidf_similarity(resume_text, job_text) * 100
    keyword_result = compute_keyword_overlap(resume_text, job.get("keywords", []))
    keyword_score = keyword_result["score"]

    final_score = round((tfidf_score * 0.70) + (keyword_score * 0.30), 2)

    return {
        "score": final_score,
        "tfidf_similarity": round(tfidf_score, 2),
        "keyword_overlap": keyword_score,
        "matched_keywords": keyword_result["matched"],
        "missing_keywords": keyword_result["missing"],
    }
