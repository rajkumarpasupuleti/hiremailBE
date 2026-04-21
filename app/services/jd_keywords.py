"""
JD text to keyword extraction and boolean query generation.

The extraction brain lives in app/data/skills_db.json.
To improve results — add section headings, prefix/suffix patterns, or skills there.
No Python changes needed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.engine.text_utils import clean_text, dedupe_preserve_order

# ── load config from skills_db.json ──────────────────────────────────────────

_DB_PATH = Path(__file__).parent.parent / "data" / "skills_db.json"

with _DB_PATH.open(encoding="utf-8") as _f:
    _DB = json.load(_f)

SECTION_HEADINGS: dict[str, str] = _DB["section_headings"]
SKILL_PREFIX_PATTERNS: tuple[str, ...] = tuple(_DB["prefix_patterns"])
SKILL_SUFFIX_WORDS: tuple[str, ...] = tuple(_DB["suffix_words"])
PHRASE_BLACKLIST: tuple[str, ...] = tuple(_DB["phrase_blacklist"])
_SKILLS_BY_DOMAIN: dict[str, list[str]] = _DB["skills"]

# Flat list for fast substring matching — longest phrases first to avoid partial overlaps
_SKILLS_FLAT: list[str] = sorted(
    [skill for skills in _SKILLS_BY_DOMAIN.values() for skill in skills],
    key=len,
    reverse=True,
)

# ── compiled regexes (stay in Python — too complex for JSON) ──────────────────

_CREDENTIAL_RE = re.compile(
    r"\b(licensed|certified)\s+([\w][\w\s\-]+?)"
    r"(?=\s*(?:preferred|required|is a plus|$|,|\.))",
    re.IGNORECASE,
)

_LEADING_FILLER_RE = re.compile(
    r"^(strong|excellent|good|solid|great|exceptional|proven|demonstrated|superior|outstanding|relevant|related|advanced|advanced-level|intermediate|entry-level|senior|junior|high)\s+",
    re.IGNORECASE,
)

_TRAILING_NOISE_RE = re.compile(
    r"\s+(is a plus|is preferred|is required|is an asset|a plus|or equivalent|or related)\s*$",
    re.IGNORECASE,
)

_LEADING_EXPERIENCE_RE = re.compile(
    r"^\d+[\+\-]?\s*(?:to\s*\d+\s*)?years?\s+(?:of\s+)?(?:recent\s+)?(?:of\s+)?",
    re.IGNORECASE,
)


_INLINE_PREFERRED_RE = re.compile(
    r"\b(preferred|is a plus|nice to have|bonus)\b", re.IGNORECASE
)

# Strip leading numbered list prefixes: "1.", "2.", "10." etc.
_NUMBERED_LIST_RE = re.compile(r"^\d+[\.\)]\s*")

# Strip parentheses and their content only when they contain non-skill noise
_PARENS_RE = re.compile(r"[()]")


# ── public entry point ────────────────────────────────────────────────────────

def extract_jd_keywords(job_description: str) -> dict:
    text = clean_text(job_description)
    lines = [
        _NUMBERED_LIST_RE.sub("", clean_text(line).strip(" -\t*•"))
        for line in text.splitlines()
        if clean_text(line).strip(" -\t*•")
    ]
    lines = [l for l in lines if l]

    if not lines:
        return {"keywords": [], "required_keywords": [], "preferred_keywords": [], "boolean_query": ""}

    sections = _extract_sections(lines)

    req_lines = sections["minimum_qualifications"] + sections["responsibilities"]
    pref_lines = sections["preferred_qualifications"]

    raw_required = _mine_lines(req_lines)
    explicit_preferred = _mine_lines(pref_lines)

    # Fallback: no section headings detected — mine all lines
    if not raw_required and not explicit_preferred:
        raw_required = _mine_lines(lines)

    # Promote keywords whose source line has an inline "preferred" marker
    inline_required, inline_preferred = _split_inline_preferred(raw_required, req_lines)

    required_keywords = inline_required
    preferred_keywords = dedupe_preserve_order(inline_preferred + explicit_preferred)

    req_set = {k.casefold() for k in required_keywords}
    preferred_keywords = [k for k in preferred_keywords if k.casefold() not in req_set]

    # Remove keywords that are word-boundary substrings of a longer keyword in the same list
    all_kws = required_keywords + preferred_keywords
    all_kws = _remove_subset_keywords(all_kws)
    req_set2 = {k.casefold() for k in required_keywords}
    required_keywords = [k for k in all_kws if k.casefold() in req_set2]
    preferred_keywords = [k for k in all_kws if k.casefold() not in req_set2]

    keywords = dedupe_preserve_order(required_keywords + preferred_keywords)
    boolean_query = _build_boolean_query(required_keywords, preferred_keywords)

    return {
        "keywords": keywords,
        "required_keywords": required_keywords,
        "preferred_keywords": preferred_keywords,
        "boolean_query": boolean_query,
    }


# ── section splitting ─────────────────────────────────────────────────────────

def _extract_sections(lines: list[str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {
        "preamble": [],
        "minimum_qualifications": [],
        "preferred_qualifications": [],
        "responsibilities": [],
    }
    # Default to preamble so boilerplate/disclaimers before any heading are never mined
    current = "preamble"
    for line in lines:
        # Try full line as heading first (heading on its own line)
        heading_key = _normalize_heading(line.rstrip(": "))
        if heading_key in SECTION_HEADINGS:
            current = SECTION_HEADINGS[heading_key]
            continue

        # Try "HEADING: content on same line" — split at first colon
        if ":" in line:
            before_colon, _, after_colon = line.partition(":")
            inline_key = _normalize_heading(before_colon)
            if inline_key in SECTION_HEADINGS:
                current = SECTION_HEADINGS[inline_key]
                remainder = after_colon.strip()
                if remainder:
                    sections[current].append(remainder)
                continue

        sections[current].append(line)
    return sections


# ── main mining dispatcher ────────────────────────────────────────────────────

def _mine_lines(lines: list[str]) -> list[str]:
    phrases: list[str] = []
    phrases.extend(_extract_prefix_phrases(lines))
    phrases.extend(_extract_suffix_phrases(lines))
    phrases.extend(_extract_credentials(lines))
    phrases.extend(_extract_from_skills_db(lines))
    return dedupe_preserve_order(phrases)


# ── strategy 1: forward prefix ────────────────────────────────────────────────

def _extract_prefix_phrases(lines: list[str]) -> list[str]:
    phrases: list[str] = []
    for line in lines:
        cleaned = clean_text(line).rstrip(".")
        lowered = cleaned.lower()
        for prefix in SKILL_PREFIX_PATTERNS:
            idx = lowered.find(prefix)
            if idx == -1:
                continue
            chunk = cleaned[idx + len(prefix):]
            chunk = re.split(r"\bor equivalent\b|\bthat\b", chunk, maxsplit=1, flags=re.IGNORECASE)[0]
            for part in re.split(r",|\band\b|\bor\b|/|\|", chunk):
                phrase = _clean_phrase(part)
                if _is_valid_phrase(phrase):
                    phrases.append(phrase)
    return phrases


# ── strategy 2: reverse suffix ────────────────────────────────────────────────

def _extract_suffix_phrases(lines: list[str]) -> list[str]:
    phrases: list[str] = []
    for line in lines:
        cleaned = clean_text(line).rstrip(".")
        lowered = cleaned.lower()
        for suffix in SKILL_SUFFIX_WORDS:
            match = re.search(rf"\b{suffix}\b", lowered)
            if not match:
                continue
            after = lowered[match.end():].strip(" ,;.()")
            if after and not re.match(r"^(in|of|with|a|an|the|is|are|preferred|required)\b", after):
                continue
            before = cleaned[: match.start()].strip(" ,;")
            if not before:
                continue
            for part in re.split(r",|\band\b|\bor\b|/|\bas well as\b", before):
                part = part.strip()
                part = _LEADING_FILLER_RE.sub("", part).strip()
                if not part:
                    continue
                # If the part alone is already a known skill, skip adding "X experience/skills"
                # — the skills DB extractor will pick it up cleanly
                if any(part.casefold() == s.casefold() for s in _SKILLS_FLAT):
                    continue
                phrase = _clean_phrase(f"{part} {suffix}")
                if _is_valid_phrase(phrase):
                    phrases.append(phrase)
            break
    return phrases


# ── strategy 3: credentials ───────────────────────────────────────────────────

def _extract_credentials(lines: list[str]) -> list[str]:
    phrases: list[str] = []
    for line in lines:
        for match in _CREDENTIAL_RE.finditer(line):
            phrase = _clean_phrase(match.group(0))
            if _is_valid_phrase(phrase):
                phrases.append(phrase)
    return phrases


# ── strategy 4: skills database matching ─────────────────────────────────────

def _extract_from_skills_db(lines: list[str]) -> list[str]:
    """
    Direct substring match against skills_db.json.
    Longest skill phrases checked first to avoid partial overlaps.
    """
    joined = " " + " ".join(lines).lower() + " "
    matched: list[str] = []
    for skill in _SKILLS_FLAT:
        needle = skill.lower()
        pattern = rf"(?<![a-z]){re.escape(needle)}(?![a-z])"
        # Exclude matches that are part of a longer word (e.g. "SQL" inside "NoSQL")
        for m in re.finditer(pattern, joined):
            surrounding = joined[max(0, m.start()-3):m.end()+3]
            if needle == "sql" and "nosql" in surrounding:
                continue
            matched.append(skill)
            break
    return matched


# ── subset deduplication ──────────────────────────────────────────────────────

def _remove_subset_keywords(keywords: list[str]) -> list[str]:
    """
    Remove any keyword that is a word-boundary substring of a longer keyword
    in the same list.
    e.g. ["Microsoft Excel", "Excel"] → ["Microsoft Excel"]
         ["print production experience", "print production"] → ["print production experience"]
    """
    result = []
    for i, kw in enumerate(keywords):
        pattern = r"(?<!\w)" + re.escape(kw.lower()) + r"(?!\w)"
        covered = any(
            i != j and kw.lower() != keywords[j].lower()
            and re.search(pattern, keywords[j].lower())
            for j in range(len(keywords))
        )
        if not covered:
            result.append(kw)
    return result


# ── inline preferred splitter ─────────────────────────────────────────────────

def _split_inline_preferred(
    keywords: list[str],
    source_lines: list[str],
) -> tuple[list[str], list[str]]:
    preferred_kw_set: set[str] = set()
    for line in source_lines:
        if not _INLINE_PREFERRED_RE.search(line):
            continue
        for kw in _mine_lines([line]):
            preferred_kw_set.add(kw.casefold())

    required = [kw for kw in keywords if kw.casefold() not in preferred_kw_set]
    preferred = [kw for kw in keywords if kw.casefold() in preferred_kw_set]
    return required, preferred


# ── boolean query builder ─────────────────────────────────────────────────────

def _build_boolean_query(required_keywords: list[str], preferred_keywords: list[str]) -> str:
    parts: list[str] = []
    if required_keywords:
        parts.append(f"({' OR '.join(required_keywords)})")
    if preferred_keywords:
        parts.append(f"({' OR '.join(preferred_keywords)})")
    return " AND ".join(parts)


# ── helpers ───────────────────────────────────────────────────────────────────

def _clean_phrase(raw: str) -> str:
    phrase = clean_text(raw).strip(". ")
    phrase = _PARENS_RE.sub("", phrase).strip()
    phrase = re.sub(r"^(the|a|an)\s+", "", phrase, flags=re.IGNORECASE)
    phrase = _TRAILING_NOISE_RE.sub("", phrase).strip()
    phrase = _LEADING_EXPERIENCE_RE.sub("", phrase).strip()
    return phrase


def _is_valid_phrase(phrase: str) -> bool:
    if not phrase:
        return False
    word_count = len(phrase.split())
    if word_count < 2 or word_count > 6:  # min 2 words — removes single-word noise like "one"
        return False
    lowered = phrase.lower()
    if any(bad in lowered for bad in PHRASE_BLACKLIST):
        return False
    return True


def _normalize_heading(value: str) -> str:
    normalized = re.sub(r"[^a-z ]", "", clean_text(value).lower())
    return re.sub(r" +", " ", normalized).strip()
