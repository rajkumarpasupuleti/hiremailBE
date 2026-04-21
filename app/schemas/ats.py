"""
Pydantic schemas for ATS scoring endpoints.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ATSScoreRequest(BaseModel):
    resume: dict[str, Any] = Field(..., description="Raw or normalized resume payload")
    job: dict[str, Any] = Field(..., description="Raw or normalized job payload")
    weights: dict[str, float] | None = Field(
        default=None,
        description="Optional custom weights that must sum to 1.0",
    )


class ATSScoreResponse(BaseModel):
    normalized_resume: dict[str, Any]
    normalized_job: dict[str, Any]
    result: dict[str, Any]


class AIJobRequest(BaseModel):
    title: str = Field(..., description="Selected job title")
    company: str = Field(default="", description="Company name for the selected job")
    location: str = Field(default="", description="Job location")
    work_mode: str = Field(default="", description="Remote, hybrid, on-site, etc")
    employment_type: str = Field(default="", description="Full-time, contract, etc")
    summary: str = Field(default="", description="Short job summary or teaser text")


class AIJobResponse(BaseModel):
    raw_ai_job: dict[str, Any]
    normalized_job: dict[str, Any]


class ATSUploadScoreResponse(BaseModel):
    grouped_resume: dict[str, Any]
    raw_resume: dict[str, Any]
    resume_quality: dict[str, Any]
    raw_ai_job: dict[str, Any]
    normalized_resume: dict[str, Any]
    normalized_job: dict[str, Any]
    result: dict[str, Any]
