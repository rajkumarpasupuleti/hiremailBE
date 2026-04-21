"""
Resume upload parsing service backed by internal parser modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import UploadFile

from app.core.config import settings
from app.parser.gemini_resume_formatter import GeminiResumeFormatter
from app.parser.pdf_extractor import PDFResumeExtractor

logger = logging.getLogger(__name__)


def parse_uploaded_resume(file: UploadFile) -> dict[str, Any]:
    """
    Parse an uploaded text-based PDF resume through the internal parser modules.
    Returns grouped resume JSON, Gemini-formatted resume JSON, and quality metadata.
    """
    logger.info(
        "Resume parser started: filename=%s content_type=%s",
        file.filename,
        file.content_type,
    )
    if file.content_type and file.content_type not in {
        "application/pdf",
        "application/octet-stream",
    }:
        raise ValueError("Only PDF resumes are supported in this flow.")

    suffix = Path(file.filename or "resume.pdf").suffix.lower()
    if suffix != ".pdf":
        raise ValueError("Only .pdf resumes are supported in this flow.")

    file.file.seek(0)
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(file.file.read())

    try:
        extractor = PDFResumeExtractor()
        lines = extractor.extract_lines(temp_path)
        logger.info("Resume parser extracted lines: filename=%s line_count=%s", file.filename, len(lines))
        quality = extractor.assess_quality(lines)
        logger.info(
            "Resume parser quality: filename=%s selectable=%s reason=%s total_chars=%s",
            file.filename,
            quality.selectable,
            quality.reason,
            quality.total_chars,
        )
        if not quality.selectable:
            logger.error(
                "Resume parser rejected PDF: filename=%s reason=%s",
                file.filename,
                quality.reason,
            )
            raise ValueError(
                "Resume PDF must be selectable text. "
                f"Parser quality check failed with reason: {quality.reason}"
            )

        grouped_resume = extractor.section_grouper.group(lines)
        logger.info(
            "Resume grouped into sections: filename=%s sections=%s",
            file.filename,
            sorted(grouped_resume.keys()),
        )
        formatter = GeminiResumeFormatter(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
        )
        raw_resume = formatter.format(grouped_resume)
        logger.info(
            "Gemini resume formatting completed: filename=%s raw_resume_keys=%s",
            file.filename,
            sorted(raw_resume.keys()),
        )

        return {
            "grouped_resume": grouped_resume,
            "raw_resume": raw_resume,
            "quality": {
                "selectable": quality.selectable,
                "reason": quality.reason,
                "total_chars": quality.total_chars,
                "alpha_chars": quality.alpha_chars,
                "alpha_ratio": quality.alpha_ratio,
                "line_count": quality.line_count,
            },
        }
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Failed to delete temporary resume file: path=%s", temp_path)
            pass
