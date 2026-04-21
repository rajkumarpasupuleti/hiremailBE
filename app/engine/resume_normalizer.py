"""
Normalize raw resume payloads into the internal schema used by the scorer.
"""

from __future__ import annotations

from datetime import date
import re

from app.engine.text_utils import clean_text, dedupe_preserve_order

EDUCATION_LEVEL_CODE_TO_ALIAS = {
    "16": "bachelors",
    "18": "masters",
}


def normalize_resume_payload(resume_payload: dict) -> dict:
    """Return a normalized resume dict for internal ATS scoring."""
    if _is_internal_resume_schema(resume_payload):
        normalized = {
            "name": clean_text(resume_payload.get("name", "")),
            "email": clean_text(resume_payload.get("email", "")),
            "phone": clean_text(resume_payload.get("phone", "")),
            "skills": dedupe_preserve_order(resume_payload.get("skills", [])),
            "experience": _normalize_internal_experience(resume_payload.get("experience", [])),
            "education": _normalize_internal_education(resume_payload.get("education", [])),
            "certifications": dedupe_preserve_order(resume_payload.get("certifications", [])),
            "summary": clean_text(resume_payload.get("summary", "")),
            "projects": _normalize_internal_projects(resume_payload.get("projects", [])),
            "_debug": {"source_format": "internal", "normalization_applied": False},
        }
        return normalized

    if _is_gemini_resume_schema(resume_payload):
        return _normalize_gemini_resume_payload(resume_payload)

    if _is_grouped_resume_schema(resume_payload):
        return _normalize_grouped_resume_payload(resume_payload)

    data = resume_payload.get("data", {})
    person = data.get("person", {})
    profiles = data.get("profiles", [])
    profile = profiles[0] if profiles else {}
    communication = person.get("communication", {})

    qualifications = _ensure_list(profile.get("qualifications", []))
    employment = _ensure_list(profile.get("employment", []))
    education_entries = _ensure_list(profile.get("education", []))
    person_competencies = _ensure_list(profile.get("personCompetency", []))

    normalized_skills = _extract_skills(qualifications)
    normalized_experience = _extract_experience(employment)
    normalized_education = _extract_education(education_entries)
    total_years = _extract_total_years(person_competencies)

    normalized = {
        "name": clean_text(person.get("name", {}).get("formattedName", "")),
        "email": clean_text(
            _pick_first(
                *[
                    item.get("address", "")
                    for item in _ensure_list(communication.get("email", []))
                    if isinstance(item, dict)
                ]
            )
        ),
        "phone": clean_text(
            _pick_first(
                *[
                    item.get("formattedNumber", "")
                    for item in _ensure_list(communication.get("phone", []))
                    if isinstance(item, dict)
                ]
            )
        ),
        "skills": normalized_skills,
        "experience": normalized_experience,
        "education": normalized_education,
        "certifications": [],
        "summary": clean_text(profile.get("executiveSummary", "")),
        "projects": [],
        "_debug": {
            "source_format": "raw_resume",
            "skill_names_from_qualifications": normalized_skills,
            "total_years_from_person_competency": total_years,
            "employment_count": len(normalized_experience),
            "education_count": len(normalized_education),
            "employment_preview": normalized_experience[:3],
            "education_preview": normalized_education,
        },
    }
    return normalized


def _is_internal_resume_schema(resume_payload: dict) -> bool:
    required_keys = {"skills", "experience", "education", "summary"}
    return required_keys.issubset(resume_payload.keys())


def _is_gemini_resume_schema(resume_payload: dict) -> bool:
    required_keys = {"skills", "education", "work_experience", "summary"}
    distinguishing_keys = {"person_name", "profile", "certificates", "personal_projects"}
    return required_keys.issubset(resume_payload.keys()) and bool(
        distinguishing_keys.intersection(resume_payload.keys())
    )


def _is_grouped_resume_schema(resume_payload: dict) -> bool:
    grouped_keys = {
        "header",
        "summary",
        "education",
        "skills",
        "work_experience",
        "personal_projects",
        "certificates",
        "languages",
        "interests",
    }
    return "header" in resume_payload and bool(grouped_keys.intersection(resume_payload.keys()))


def _normalize_gemini_resume_payload(resume_payload: dict) -> dict:
    skills = _extract_gemini_skills(_ensure_list(resume_payload.get("skills", [])))
    experience = _extract_gemini_experience(_ensure_list(resume_payload.get("work_experience", [])))
    education = _extract_gemini_education(_ensure_list(resume_payload.get("education", [])))
    certifications = _extract_named_values(_ensure_list(resume_payload.get("certificates", [])))
    projects = _extract_gemini_projects(_ensure_list(resume_payload.get("personal_projects", [])))
    summary = _combine_text_fields(
        clean_text(resume_payload.get("summary", "")),
        clean_text(resume_payload.get("profile", "")),
    )

    return {
        "name": clean_text(resume_payload.get("person_name", "")),
        "email": clean_text(resume_payload.get("email", "")),
        "phone": clean_text(resume_payload.get("phone", "")),
        "skills": skills,
        "experience": experience,
        "education": education,
        "certifications": certifications,
        "summary": summary,
        "projects": projects,
        "_debug": {
            "source_format": "gemini_resume",
            "skill_count": len(skills),
            "employment_count": len(experience),
            "education_count": len(education),
            "certification_count": len(certifications),
            "project_count": len(projects),
        },
    }


def _normalize_grouped_resume_payload(resume_payload: dict) -> dict:
    header = resume_payload.get("header", {})
    if not isinstance(header, dict):
        header = {}

    skills = dedupe_preserve_order(_ensure_list(resume_payload.get("skills", [])))
    experience_lines = dedupe_preserve_order(_ensure_list(resume_payload.get("work_experience", [])))
    education_lines = dedupe_preserve_order(_ensure_list(resume_payload.get("education", [])))
    summary = _combine_text_fields(
        clean_text(header.get("summary", "")),
        clean_text(" ".join(_ensure_list(resume_payload.get("summary", [])))),
        clean_text(header.get("profile", "")),
    )

    experience = [
        {
            "title": "",
            "company": "",
            "years": 0.0,
            "description": clean_text(line),
        }
        for line in experience_lines
        if clean_text(line)
    ]

    education = [
        {
            "degree": clean_text(line),
            "degree_type": _infer_degree_type(clean_text(line)),
            "field": "",
            "institution": "",
            "year": _extract_year(line),
        }
        for line in education_lines
        if clean_text(line)
    ]

    certifications = dedupe_preserve_order(_ensure_list(resume_payload.get("certificates", [])))
    projects = [
        {"name": "", "description": clean_text(line)}
        for line in _ensure_list(resume_payload.get("personal_projects", []))
        if clean_text(line)
    ]

    return {
        "name": clean_text(header.get("person_name", "")),
        "email": clean_text(header.get("email", "")),
        "phone": clean_text(header.get("phone", "")),
        "skills": skills,
        "experience": experience,
        "education": education,
        "certifications": certifications,
        "summary": summary,
        "projects": projects,
        "_debug": {
            "source_format": "grouped_resume",
            "skill_count": len(skills),
            "employment_count": len(experience),
            "education_count": len(education),
            "normalization_applied": True,
        },
    }


def _extract_skills(qualifications: list) -> list[str]:
    skills = []
    for qualification in qualifications:
        if not isinstance(qualification, dict):
            continue
        evidence = qualification.get("competencyEvidence", {}).get("descriptions", [])
        if evidence:
            skills.extend(str(item) for item in evidence if item)
            continue
        competency_name = qualification.get("competencyName", "")
        if competency_name:
            skills.append(str(competency_name))
    return dedupe_preserve_order(skills)


def _extract_experience(employment: list) -> list[dict]:
    items = []
    for item in employment:
        if not isinstance(item, dict):
            continue
        title = clean_text(
            _pick_first(
                *[
                    history.get("title", "")
                    for history in _ensure_list(item.get("positionHistories", []))
                    if isinstance(history, dict)
                ]
            )
        )
        raw_role_descriptions = [
            clean_text(description)
            for history in _ensure_list(item.get("positionHistories", []))
            if isinstance(history, dict)
            for description in _ensure_list(history.get("descriptions", []))
            if description
        ]
        company = clean_text(item.get("organization", {}).get("name", ""))
        years = _extract_employment_years(item)
        start = clean_text(item.get("start", ""))
        end = clean_text(item.get("end", "")) or ("Present" if item.get("current") else "")
        description_parts = [part for part in [title, f"at {company}" if company else "", _format_date_range(start, end)] if part]
        description_text = clean_text("\n\n".join(part for part in raw_role_descriptions if part))
        items.append(
            {
                "title": title,
                "company": company,
                "years": years,
                "description": clean_text("\n\n".join(part for part in [" ".join(description_parts), description_text] if part)),
            }
        )
    return items


def _extract_gemini_skills(skills: list) -> list[str]:
    values = []
    for skill in skills:
        if isinstance(skill, str):
            values.append(skill)
        elif isinstance(skill, dict):
            values.extend(
                str(value)
                for key, value in skill.items()
                if key in {"name", "skill", "value"} and value
            )
    return dedupe_preserve_order(values)


def _extract_gemini_experience(work_experience: list) -> list[dict]:
    items = []
    for item in work_experience:
        if not isinstance(item, dict):
            continue

        title = clean_text(item.get("title", ""))
        company = clean_text(item.get("company", ""))
        start = clean_text(item.get("start_date", ""))
        end = clean_text(item.get("end_date", ""))
        location = clean_text(item.get("location", ""))
        detail = clean_text(item.get("description", ""))
        years = _calculate_years_from_dates(start, end)

        meta_parts = [
            title,
            f"at {company}" if company else "",
            f"in {location}" if location else "",
            _format_date_range(start, end or "Present"),
        ]
        description = _combine_text_fields(" ".join(part for part in meta_parts if part), detail)

        items.append(
            {
                "title": title,
                "company": company,
                "years": years,
                "description": description,
            }
        )
    return items


def _extract_education(education_entries: list) -> list[dict]:
    items = []
    for entry in education_entries:
        if not isinstance(entry, dict):
            continue

        degree_names = []
        degree_codes = []
        field_names = []
        for degree in _ensure_list(entry.get("educationDegrees", [])):
            if not isinstance(degree, dict):
                continue
            if degree.get("name"):
                degree_names.append(degree.get("name", ""))
            degree_codes.extend(code for code in _ensure_list(degree.get("codes", [])) if code)
            for specialization in _ensure_list(degree.get("specializations", [])):
                if not isinstance(specialization, dict):
                    continue
                name = specialization.get("name", "")
                if name:
                    field_names.append(name)

        level_codes = [
            clean_text(level.get("id", {}).get("value", ""))
            for level in _ensure_list(entry.get("educationLevelCodes", []))
            if isinstance(level, dict)
        ]

        degree = clean_text(_pick_first(*degree_names, *degree_codes))
        field = clean_text(_pick_first(*field_names))
        degree_type = clean_text(_pick_first(*[EDUCATION_LEVEL_CODE_TO_ALIAS.get(code, "") for code in level_codes]))
        institution = clean_text(entry.get("institution", {}).get("name", ""))
        year = _extract_year(_pick_first(entry.get("end"), entry.get("start")))

        items.append(
            {
                "degree": degree or field or institution,
                "degree_type": degree_type,
                "field": field,
                "institution": institution,
                "year": year,
            }
        )
    return items


def _extract_gemini_education(education_entries: list) -> list[dict]:
    items = []
    for entry in education_entries:
        if not isinstance(entry, dict):
            continue

        degree = clean_text(entry.get("degree", ""))
        institution = clean_text(entry.get("institution", ""))
        level = clean_text(entry.get("level", ""))
        description = clean_text(entry.get("description", ""))
        start = clean_text(entry.get("start_date", ""))
        end = clean_text(entry.get("end_date", ""))

        items.append(
            {
                "degree": degree or institution or description,
                "degree_type": clean_text(_pick_first(_normalize_degree_level(level), _infer_degree_type(degree))),
                "field": "",
                "institution": institution,
                "year": _extract_year(_pick_first(end, start)),
            }
        )
    return items


def _extract_total_years(person_competencies: list) -> float:
    for item in person_competencies:
        if not isinstance(item, dict):
            continue
        competency_name = clean_text(item.get("competencyName", "")).lower()
        if competency_name == "total years of experience":
            return _safe_float(item.get("experienceMeasure", {}).get("value", 0))
    return 0.0


def _extract_employment_years(item: dict) -> float:
    years = _safe_float(item.get("relatedCompetencies", {}).get("experienceMeasure", {}).get("value", None), default=None)
    if years is not None:
        return years

    start = clean_text(item.get("start", ""))
    end = clean_text(item.get("end", "")) if not item.get("current") else str(date.today())
    if start and end:
        try:
            start_year, start_month, _ = (int(part) for part in start.split("-"))
            end_year, end_month, _ = (int(part) for part in end.split("-"))
            total_months = (end_year - start_year) * 12 + (end_month - start_month)
            return round(max(total_months, 0) / 12, 2)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _normalize_internal_experience(experience: list) -> list[dict]:
    items = []
    for item in experience:
        if not isinstance(item, dict):
            continue
        items.append(
            {
                "title": clean_text(item.get("title", "")),
                "company": clean_text(item.get("company", "")),
                "years": _safe_float(item.get("years", 0)),
                "description": clean_text(item.get("description", "")),
            }
        )
    return items


def _normalize_internal_education(education: list) -> list[dict]:
    items = []
    for item in education:
        if not isinstance(item, dict):
            continue
        items.append(
            {
                "degree": clean_text(item.get("degree", "")),
                "degree_type": clean_text(item.get("degree_type", "")),
                "field": clean_text(item.get("field", "")),
                "institution": clean_text(item.get("institution", "")),
                "year": item.get("year"),
            }
        )
    return items


def _normalize_internal_projects(projects: list) -> list[dict]:
    items = []
    for item in projects:
        if not isinstance(item, dict):
            continue
        items.append({"name": clean_text(item.get("name", "")), "description": clean_text(item.get("description", ""))})
    return items


def _extract_named_values(items: list) -> list[str]:
    values = []
    for item in items:
        if isinstance(item, str):
            values.append(item)
            continue
        if not isinstance(item, dict):
            continue
        values.append(
            str(
                _pick_first(
                    item.get("name", ""),
                    item.get("title", ""),
                    item.get("certificate", ""),
                    item.get("description", ""),
                )
            )
        )
    return dedupe_preserve_order(values)


def _extract_gemini_projects(projects: list) -> list[dict]:
    items = []
    for item in projects:
        if not isinstance(item, dict):
            continue
        name = clean_text(item.get("name", ""))
        description = _combine_text_fields(clean_text(item.get("date", "")), clean_text(item.get("description", "")))
        items.append({"name": name, "description": description})
    return items


def _extract_year(value: object) -> int | None:
    text = clean_text(value)
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    return None


def _parse_date_parts(value: str) -> tuple[int, int] | None:
    text = clean_text(value)
    if not text:
        return None

    if text.lower() == "present":
        today = date.today()
        return today.year, today.month

    match = re.match(r"^(\d{4})(?:-(\d{1,2}))?", text)
    if not match:
        return None

    year = int(match.group(1))
    month = int(match.group(2) or 1)
    month = max(1, min(month, 12))
    return year, month


def _calculate_years_from_dates(start: str, end: str) -> float:
    start_parts = _parse_date_parts(start)
    end_parts = _parse_date_parts(end or "Present")
    if not start_parts or not end_parts:
        return 0.0

    start_year, start_month = start_parts
    end_year, end_month = end_parts
    total_months = (end_year - start_year) * 12 + (end_month - start_month)
    return round(max(total_months, 0) / 12, 2)


def _format_date_range(start: str, end: str) -> str:
    if start and end:
        return f"({start} to {end})"
    if start:
        return f"(from {start})"
    return ""


def _combine_text_fields(*values: str) -> str:
    cleaned_values = [clean_text(value) for value in values if clean_text(value)]
    return clean_text("\n\n".join(cleaned_values))


def _normalize_degree_level(level: object) -> str:
    normalized = clean_text(level).lower()
    if normalized in {"bachelor", "bachelors", "undergraduate", "ug"}:
        return "bachelors"
    if normalized in {"master", "masters", "postgraduate", "pg"}:
        return "masters"
    if normalized in {"doctorate", "doctoral", "phd"}:
        return "doctorate"
    if normalized in {"associate", "diploma"}:
        return "diploma"
    if normalized in {"high school", "secondary"}:
        return "high_school"
    return clean_text(level)


def _infer_degree_type(value: str) -> str:
    normalized = clean_text(value).lower()
    if re.search(r"\b(phd|doctorate|doctoral|dphil)\b", normalized):
        return "doctorate"
    if re.search(r"\b(master|masters|mba|mtech|m tech|msc|m sc|mca|ms)\b", normalized):
        return "masters"
    if re.search(r"\b(bachelor|bachelors|btech|b tech|bsc|b sc|ba|b a|bca|bba|be|b e)\b", normalized):
        return "bachelors"
    if re.search(r"\b(diploma|associate|associates|polytechnic)\b", normalized):
        return "diploma"
    if re.search(r"\b(high school|secondary|hsc|ssc|12th|10th)\b", normalized):
        return "high_school"
    return ""


def _safe_float(value: object, default: float | None = 0.0) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pick_first(*values: object) -> object:
    for value in values:
        if value:
            return value
    return ""


def _ensure_list(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
