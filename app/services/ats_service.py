"""
Service layer for ATS scoring.
"""

from __future__ import annotations

import logging

from fastapi import UploadFile

from app.engine.job_normalizer import normalize_job_payload
from app.engine.resume_normalizer import normalize_resume_payload
from app.engine.scorer import compute_final_score
from app.services.gemini_job_service import generate_job_requirements
from app.services.resume_parser_service import parse_uploaded_resume

logger = logging.getLogger(__name__)


def score_resume_against_job(resume: dict, job: dict, weights: dict | None = None) -> dict:
    """
    Normalize the incoming payloads and compute the ATS score using the
    self-contained engine inside the FastAPI application.
    """
    logger.info("Normalizing resume and job payloads for /ats/score")
    normalized_resume = normalize_resume_payload(resume)
    normalized_job = normalize_job_payload(job)
    logger.info(
        "Normalized payloads: resume_skills=%s resume_experience=%s job_required_skills=%s",
        len(normalized_resume.get("skills", [])),
        len(normalized_resume.get("experience", [])),
        len(normalized_job.get("required_skills", [])),
    )
    result = compute_final_score(normalized_resume, normalized_job, weights=weights)
    logger.info(
        "Computed ATS score for /ats/score: final_score=%s grade=%s",
        result.get("final_score"),
        result.get("grade"),
    )

    return {
        "normalized_resume": normalized_resume,
        "normalized_job": normalized_job,
        "result": result,
    }


def generate_ai_job_payload(job_input: dict) -> dict:
    """
    Generate ATS-friendly job requirements from lightweight job metadata
    and normalize them into the scorer's internal schema.
    """
    logger.info(
        "Generating AI job payload: title=%s company=%s",
        job_input.get("title", ""),
        job_input.get("company", ""),
    )
    raw_ai_job = generate_job_requirements(job_input)
    normalized_job = normalize_job_payload(raw_ai_job)
    logger.info(
        "Generated and normalized AI job: normalized_title=%s required_skills=%s preferred_skills=%s",
        normalized_job.get("title", ""),
        len(normalized_job.get("required_skills", [])),
        len(normalized_job.get("preferred_skills", [])),
    )
    return {
        "raw_ai_job": raw_ai_job,
        "normalized_job": normalized_job,
    }


def score_uploaded_resume_against_ai_job(
    file: UploadFile,
    job_input: dict,
    weights: dict | None = None,
) -> dict:
    """
    Parse an uploaded resume PDF, generate AI job requirements from lightweight job
    metadata, normalize both, and compute the ATS score.
    """
    logger.info(
        "Starting upload-score flow: filename=%s title=%s",
        file.filename,
        job_input.get("title", ""),
    )
    resume_payload = parse_uploaded_resume(file)
    raw_resume = resume_payload["raw_resume"]
    grouped_resume = resume_payload["grouped_resume"]
    quality = resume_payload["quality"]
    logger.info(
        "Resume parsed: selectable=%s quality_reason=%s grouped_sections=%s raw_resume_keys=%s",
        quality.get("selectable"),
        quality.get("reason"),
        len(grouped_resume.keys()),
        sorted(raw_resume.keys()),
    )

    normalized_resume = normalize_resume_payload(raw_resume)
    logger.info(
        "Resume normalized: name=%s skills=%s experience=%s",
        normalized_resume.get("name", ""),
        len(normalized_resume.get("skills", [])),
        len(normalized_resume.get("experience", [])),
    )

    job_payload = generate_ai_job_payload(job_input)
    raw_ai_job = job_payload["raw_ai_job"]
    normalized_job = job_payload["normalized_job"]
    logger.info(
        "AI job ready: raw_keys=%s normalized_title=%s required_skills=%s",
        sorted(raw_ai_job.keys()),
        normalized_job.get("title", ""),
        len(normalized_job.get("required_skills", [])),
    )

    result = compute_final_score(normalized_resume, normalized_job, weights=weights)
    logger.info(
        "Upload-score completed: filename=%s final_score=%s grade=%s",
        file.filename,
        result.get("final_score"),
        result.get("grade"),
    )

    return {
        "grouped_resume": grouped_resume,
        "raw_resume": raw_resume,
        "resume_quality": quality,
        "raw_ai_job": raw_ai_job,
        "normalized_resume": normalized_resume,
        "normalized_job": normalized_job,
        "result": result,
    }
