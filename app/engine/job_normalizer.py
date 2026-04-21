"""
Normalize raw job payloads into the internal schema used by the scorer.
"""

from __future__ import annotations

import json
import re

from app.engine.text_utils import clean_text, dedupe_preserve_order

SECTION_HEADINGS = {
    "requirements": "requirements",
    "minimum qualifications": "minimum_qualifications",
    "preferred qualifications": "preferred_qualifications",
    "work conditions": "work_conditions",
}

SKILL_PREFIX_PATTERNS = (
    r"experience in ",
    r"experience using ",
    r"experience producing ",
    r"experience designing ",
    r"strong skills in ",
    r"knowledge of ",
    r"familiarity with ",
    r"adept at ",
)

PHRASE_BLACKLIST = (
    "related fields",
    "equivalent combination of education and experience",
    "modern learners",
    "at once",
    "the ability to travel",
)


def normalize_job_payload(job_payload: dict) -> dict:
    """Return a normalized job dict for internal ATS scoring."""
    if _is_internal_job_schema(job_payload):
        normalized = {
            "title": clean_text(job_payload.get("title", "")),
            "company": clean_text(job_payload.get("company", "")),
            "required_skills": dedupe_preserve_order(job_payload.get("required_skills", [])),
            "preferred_skills": dedupe_preserve_order(job_payload.get("preferred_skills", [])),
            "experience_years_required": _safe_int(job_payload.get("experience_years_required", 0)),
            "education_required": clean_text(job_payload.get("education_required", "")),
            "keywords": dedupe_preserve_order(job_payload.get("keywords", [])),
            "description": clean_text(job_payload.get("description", "")),
        }
        normalized["_debug"] = {"source_format": "internal", "normalization_applied": False}
        return normalized

    if _is_gemini_job_schema(job_payload):
        return _normalize_gemini_job_payload(job_payload)

    value = job_payload.get("Value", {})
    parsed_document = _load_parsed_document(value.get("ParsedDocument"))
    sovren = parsed_document.get("SovrenData", {})

    taxonomy_skills = _extract_taxonomy_skills(sovren.get("SkillsTaxonomyOutput", []))
    required_taxonomy_skills = [item["name"] for item in taxonomy_skills if item["required"]]
    optional_taxonomy_skills = [
        item["name"] for item in taxonomy_skills if not item["required"] and item["exists_in_text"]
    ]

    job_description = clean_text(sovren.get("JobDescription", ""))
    job_requirements = clean_text(sovren.get("JobRequirements", ""))
    source_text = clean_text(sovren.get("SourceText") or value.get("Text", ""))
    sections = _extract_sections(job_requirements)

    minimum_qualification_lines = sections.get("minimum_qualifications", [])
    preferred_qualification_lines = sections.get("preferred_qualifications", [])

    required_text_skills = _extract_compact_skill_phrases(minimum_qualification_lines)
    preferred_text_skills = _extract_compact_skill_phrases(preferred_qualification_lines)
    fulltext_keywords = _extract_fulltext_keywords(sovren.get("FulltextKeywords", []))

    description_parts = [part for part in [job_description, job_requirements] if part]

    normalized = {
        "title": clean_text(
            _pick_first(
                sovren.get("JobTitles", {}).get("MainJobTitle"),
                _pick_first(*_ensure_list(sovren.get("JobTitles", {}).get("JobTitle", []))),
            )
        ),
        "company": clean_text(
            _pick_first(
                sovren.get("EmployerNames", {}).get("MainEmployerName"),
                _pick_first(*_ensure_list(sovren.get("EmployerNames", {}).get("EmployerName", []))),
            )
        ),
        "required_skills": dedupe_preserve_order(required_taxonomy_skills + required_text_skills),
        "preferred_skills": dedupe_preserve_order(preferred_text_skills + optional_taxonomy_skills),
        "experience_years_required": _safe_int(sovren.get("MinimumYears", 0)),
        "education_required": clean_text(
            sovren.get("RequiredDegree")
            or _pick_first(
                *[
                    degree.get("DegreeType", "")
                    for degree in _ensure_list(sovren.get("Education", {}).get("Degree", []))
                    if isinstance(degree, dict)
                ]
            )
        ),
        "keywords": dedupe_preserve_order(
            required_taxonomy_skills
            + optional_taxonomy_skills
            + required_text_skills
            + preferred_text_skills
            + fulltext_keywords
        ),
        "description": "\n\n".join(description_parts),
        "_debug": {
            "source_format": "sovren",
            "title_source": clean_text(sovren.get("JobTitles", {}).get("MainJobTitle", "")),
            "company_source": clean_text(sovren.get("EmployerNames", {}).get("MainEmployerName", "")),
            "education_source": clean_text(sovren.get("RequiredDegree", "")),
            "minimum_years_source": clean_text(sovren.get("MinimumYears", "")),
            "required_skills_from_taxonomy": required_taxonomy_skills,
            "optional_skills_from_taxonomy": optional_taxonomy_skills,
            "required_skill_phrases_from_text": required_text_skills,
            "preferred_skill_phrases_from_text": preferred_text_skills,
            "minimum_qualification_lines": minimum_qualification_lines,
            "preferred_qualification_lines": preferred_qualification_lines,
            "fulltext_keywords": fulltext_keywords,
            "source_text_preview": source_text[:800],
        },
    }
    return normalized


def _is_internal_job_schema(job_payload: dict) -> bool:
    required_keys = {
        "title",
        "required_skills",
        "preferred_skills",
        "experience_years_required",
        "education_required",
        "keywords",
        "description",
    }
    return required_keys.issubset(job_payload.keys())


def _is_gemini_job_schema(job_payload: dict) -> bool:
    required_keys = {
        "job_title",
        "required_skills",
        "preferred_skills",
        "keywords",
        "experience_years_required",
        "education_required",
    }
    return required_keys.issubset(job_payload.keys())


def _normalize_gemini_job_payload(job_payload: dict) -> dict:
    title = clean_text(job_payload.get("job_title", ""))
    company = clean_text(job_payload.get("company_name", ""))
    required_skills = dedupe_preserve_order(_ensure_list(job_payload.get("required_skills", [])))
    preferred_skills = dedupe_preserve_order(_ensure_list(job_payload.get("preferred_skills", [])))
    responsibilities = dedupe_preserve_order(_ensure_list(job_payload.get("responsibilities", [])))
    keywords = dedupe_preserve_order(
        _ensure_list(job_payload.get("keywords", []))
        + required_skills
        + preferred_skills
    )
    description = "\n\n".join(
        part
        for part in [
            clean_text(job_payload.get("job_summary", "")),
            "\n".join(f"- {item}" for item in responsibilities if item),
        ]
        if part
    )

    return {
        "title": title,
        "company": company,
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "experience_years_required": _safe_int(job_payload.get("experience_years_required", 0)),
        "education_required": clean_text(job_payload.get("education_required", "")),
        "keywords": keywords,
        "description": clean_text(description),
        "_debug": {
            "source_format": "gemini_job",
            "job_title_source": title,
            "company_source": company,
            "seniority_source": clean_text(job_payload.get("seniority", "")),
            "location_source": clean_text(job_payload.get("location", "")),
            "work_mode_source": clean_text(job_payload.get("work_mode", "")),
            "employment_type_source": clean_text(job_payload.get("employment_type", "")),
            "responsibility_count": len(responsibilities),
            "required_skill_count": len(required_skills),
            "preferred_skill_count": len(preferred_skills),
            "keyword_count": len(keywords),
        },
    }


def _load_parsed_document(parsed_document: object) -> dict:
    if isinstance(parsed_document, dict):
        return parsed_document
    if isinstance(parsed_document, str) and parsed_document.strip():
        try:
            return json.loads(parsed_document)
        except json.JSONDecodeError:
            return {}
    return {}


def _extract_taxonomy_skills(node: object) -> list[dict]:
    skills = []
    if isinstance(node, dict):
        if "@name" in node and "@required" in node:
            skills.append(
                {
                    "name": clean_text(node.get("@name", "")),
                    "required": _to_bool(node.get("@required")),
                    "exists_in_text": _to_bool(node.get("@existsInText", True)),
                }
            )
        for value in node.values():
            skills.extend(_extract_taxonomy_skills(value))
        return skills
    if isinstance(node, list):
        for item in node:
            skills.extend(_extract_taxonomy_skills(item))
    return skills


def _extract_fulltext_keywords(items: list) -> list[str]:
    keywords = []
    for item in _ensure_list(items):
        if not isinstance(item, dict):
            continue
        keyword = item.get("fulltext_keywords", {}).get("value", "")
        keywords.append(clean_text(keyword))
    return dedupe_preserve_order(keywords)


def _extract_sections(text: str) -> dict[str, list[str]]:
    sections = {section_name: [] for section_name in set(SECTION_HEADINGS.values())}
    current_section = "minimum_qualifications"
    for raw_line in text.splitlines():
        line = clean_text(raw_line).strip(" -\t*")
        if not line:
            continue
        heading_key = _normalize_heading(line.rstrip(":"))
        if heading_key in SECTION_HEADINGS:
            current_section = SECTION_HEADINGS[heading_key]
            continue
        sections[current_section].append(line)
    return sections


def _extract_compact_skill_phrases(lines: list[str]) -> list[str]:
    phrases = []
    for line in lines:
        cleaned_line = clean_text(line).rstrip(".")
        lowered = cleaned_line.lower()
        for prefix in SKILL_PREFIX_PATTERNS:
            index = lowered.find(prefix)
            if index == -1:
                continue
            chunk = cleaned_line[index + len(prefix) :]
            chunk = re.split(r"\bor equivalent\b|\bthat\b", chunk, maxsplit=1, flags=re.IGNORECASE)[0]
            for part in re.split(r",|\band\b|\bor\b", chunk):
                phrase = clean_text(part).strip(". ")
                phrase = re.sub(r"^(the|a|an)\s+", "", phrase, flags=re.IGNORECASE)
                word_count = len(phrase.split())
                if not phrase or word_count > 6:
                    continue
                if any(bad in phrase.lower() for bad in PHRASE_BLACKLIST):
                    continue
                phrases.append(phrase)
    return dedupe_preserve_order(phrases)


def _normalize_heading(value: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", clean_text(value).lower()).strip()


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


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
