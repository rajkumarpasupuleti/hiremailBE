"""
ATS scoring endpoints.
"""

import logging

from fastapi import APIRouter, File, Form, UploadFile

from app.schemas.ats import (
    AIJobRequest,
    AIJobResponse,
    ATSScoreRequest,
    ATSScoreResponse,
    ATSUploadScoreResponse,
)
from app.schemas.jd_keywords import JDKeywordRequest, JDKeywordResponse
from app.services.ats_service import (
    generate_ai_job_payload,
    score_resume_against_job,
    score_uploaded_resume_against_ai_job,
)
from app.services.jd_keywords import extract_jd_keywords
from app.services.jd_keywords_v2 import extract_jd_keywords_v2
from app.services.jd_keywords_v3 import extract_jd_keywords_v3

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ats", tags=["ats"])

weights_default = {  
    "skills": 0.45,
    "experience": 0.20,
    "education": 0.10,
    "keywords": 0.25,
}


@router.post("/generate-job", response_model=AIJobResponse)
def generate_job(request: AIJobRequest) -> AIJobResponse:
    logger.info(
        "POST /ats/generate-job started: title=%s company=%s location=%s",
        request.title,
        request.company,
        request.location,
    )
    try:
        payload = generate_ai_job_payload(
            {
                "title": request.title,
                "company": request.company,
                "location": request.location,
                "work_mode": request.work_mode,
                "employment_type": request.employment_type,
                "summary": request.summary,
            }
        )
        logger.info(
            "POST /ats/generate-job completed: normalized_title=%s required_skills=%s preferred_skills=%s",
            payload["normalized_job"].get("title", ""),
            len(payload["normalized_job"].get("required_skills", [])),
            len(payload["normalized_job"].get("preferred_skills", [])),
        )
        return AIJobResponse(**payload)
    except Exception:
        logger.exception(
            "POST /ats/generate-job failed: title=%s company=%s",
            request.title,
            request.company,
        )
        raise


@router.post("/upload-score", response_model=ATSUploadScoreResponse)
def upload_score(
    resume_file: UploadFile = File(...),
    title: str = Form(...),
    company: str = Form(""),
    location: str = Form(""),
    work_mode: str = Form(""),
    employment_type: str = Form(""),
    summary: str = Form(""),
) -> ATSUploadScoreResponse:
    logger.info(
        "POST /ats/upload-score started: filename=%s title=%s company=%s",
        resume_file.filename,
        title,
        company,
    )
    try:
        payload = score_uploaded_resume_against_ai_job(
            file=resume_file,
            job_input={
                "title": title,
                "company": company,
                "location": location,
                "work_mode": work_mode,
                "employment_type": employment_type,
                "summary": summary,
            },
            weights=weights_default,
        )
        logger.info(
            "POST /ats/upload-score completed: filename=%s score=%s grade=%s",
            resume_file.filename,
            payload["result"].get("final_score"),
            payload["result"].get("grade"),
        )
        return ATSUploadScoreResponse(**payload)
    except Exception:
        logger.exception(
            "POST /ats/upload-score failed: filename=%s title=%s",
            resume_file.filename,
            title,
        )
        raise


@router.post("/score", response_model=ATSScoreResponse)
def score_ats(request: ATSScoreRequest) -> ATSScoreResponse:
    logger.info("POST /ats/score started")
    try:
        payload = score_resume_against_job(
            resume=request.resume,
            job=request.job,
            weights=request.weights or weights_default,
        )
        logger.info(
            "POST /ats/score completed: score=%s grade=%s",
            payload["result"].get("final_score"),
            payload["result"].get("grade"),
        )
        return ATSScoreResponse(**payload)
    except Exception:
        logger.exception("POST /ats/score failed")
        raise


@router.post("/jd-keywords-v3", response_model=JDKeywordResponse)
def jd_keywords_v3_endpoint(request: JDKeywordRequest) -> JDKeywordResponse:
    logger.info("POST /ats/jd-keywords-v3 started")
    try:
        payload = extract_jd_keywords_v3(request.job_description)
        logger.info("POST /ats/jd-keywords-v3 completed: keywords=%s", len(payload.get("keywords", [])))
        return JDKeywordResponse(**payload)
    except Exception:
        logger.exception("POST /ats/jd-keywords-v3 failed")
        raise


@router.post("/jd-keywords-v2", response_model=JDKeywordResponse)
def jd_keywords_v2_endpoint(request: JDKeywordRequest) -> JDKeywordResponse:
    logger.info("POST /ats/jd-keywords-v2 started")
    try:
        payload = extract_jd_keywords_v2(request.job_description)
        logger.info("POST /ats/jd-keywords-v2 completed: keywords=%s", len(payload.get("keywords", [])))
        return JDKeywordResponse(**payload)
    except Exception:
        logger.exception("POST /ats/jd-keywords-v2 failed")
        raise


@router.post("/jd-keywords", response_model=JDKeywordResponse)
def jd_keywords(request: JDKeywordRequest) -> JDKeywordResponse:
    logger.info("POST /ats/jd-keywords started")
    try:
        payload = extract_jd_keywords(request.job_description)
        logger.info(
            "POST /ats/jd-keywords completed: keywords=%s required=%s preferred=%s",
            len(payload.get("keywords", [])),
            len(payload.get("required_keywords", [])),
            len(payload.get("preferred_keywords", [])),
        )
        return JDKeywordResponse(**payload)
    except Exception:
        logger.exception("POST /ats/jd-keywords failed")
        raise
