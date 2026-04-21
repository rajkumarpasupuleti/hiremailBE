"""
JD keyword extraction V2 — stop-word elimination approach.

Flow:
  1. Detect section headings (reused from v1)
  2. Tokenize section text word by word
  3. Remove English stop words (NLTK) + domain noise words
  4. What remains = keywords

No prefix/suffix patterns. No skills DB required.
"""

from __future__ import annotations

import re

import nltk
from nltk.corpus import stopwords

from app.engine.text_utils import clean_text, dedupe_preserve_order
from app.services.jd_keywords import _NUMBERED_LIST_RE, _PARENS_RE, _SKILLS_FLAT, _extract_sections

nltk.download("stopwords", quiet=True)

# ── stop word sets ─────────────────────────────────────────────────────────────

_NLTK_STOP: set[str] = set(stopwords.words("english"))

# Domain-specific noise — words common in JD sentences that are NOT skills
_DOMAIN_NOISE: set[str] = {
    # experience / time
    "experience", "experiences", "experienced",
    "years", "year", "months", "month",
    # action verbs
    "working", "work", "worked",
    "building", "build", "built",
    "using", "use", "used",
    "developing", "develop", "developed", "development",
    "managing", "manage", "managed", "management",
    "performing", "perform", "maintaining", "maintain",
    "delivering", "deliver", "delivery",
    "creating", "create", "supporting", "support",
    "ensuring", "ensure", "implementing", "implement",
    # adjectives / qualifiers
    "strong", "excellent", "good", "great", "solid", "proven",
    "advanced", "intermediate", "basic", "senior", "junior",
    "highly", "fast", "paced", "high", "low", "large", "small",
    "progressive", "related", "relevant", "various", "multiple",
    "current", "existing", "new", "emerging",
    # nouns that describe context, not skills
    "knowledge", "understanding", "familiarity", "proficiency",
    "skills", "skill", "skilled", "ability", "abilities",
    "requirements", "requirement", "qualifications", "qualification",
    "responsibilities", "responsibility",
    "degree", "bachelor", "bachelors", "masters", "equivalent",
    "field", "fields", "areas", "area", "domain",
    "team", "teams", "member", "members", "staff",
    "role", "position", "job", "title", "candidate", "candidates",
    "developer", "engineer", "specialist", "analyst", "manager", "coordinator",
    "technical", "Technical",
    "company", "organization", "business", "environment",
    "following", "including", "includes", "plus",
    "minimum", "maximum", "least", "preferred", "required", "must",
    "systems", "system", "applications", "application",
    "platform", "platforms", "tools", "tool",
    "technologies", "technology", "solutions", "solution",
    "services", "service", "products", "product",
    "please", "apply", "join", "seeking", "hiring",
    "clinical", "setting", "practice", "certification", "certifications",
    "proficient", "demonstrated", "utilize", "utilizes", "leveraging",
    "awareness", "understanding", "exposure",
    "cross", "functional", "stakeholders", "stakeholder",
    "suite", "Suite", "administrative", "executives", "executive",
    "handle", "Handle", "process", "Process", "coordinate", "Coordinate",
    "correspondence", "documents", "document", "contracts", "contract",
    "calendars", "calendar", "records", "record",
    "activities", "activity", "tasks", "task", "projects", "project",
    "procedures", "procedure", "processes", "operations", "operation",
    "collaborative", "organized", "oriented", "detail",
    "progressive", "subject", "matter", "expertise",
    "like", "type", "types", "based",
    # single generic words that slip through
    "Messaging", "messaging", "Communication", "communication",
    "Databases", "databases", "Database", "database",
    "adapting", "adapt", "Agentic", "agentic",
    "web", "Web", "sql", "Sql",
    "Apache", "apache", "Solar", "solar",
    "tools", "Tools", "emerging", "Emerging",
    "Codex", "codex",
    # job roles
    "developer", "Developer", "engineer", "Engineer",
    "specialist", "analyst", "architect", "architects",
    "testers", "tester", "leads", "Leads",
    # qualifiers
    "analytical", "able", "tackle", "complex", "issue", "hurdles",
    "deep", "Deep", "proper", "quality", "consistency", "relevance", "accuracy",
    "hands-on",
    # pay / logistics
    "pay", "Pay", "range", "Range", "hr", "w2", "W2",
    "on-site", "days", "week", "hours", "Hours", "flexible",
    "ot", "OT", "overtime", "pre-approval", "top", "TOP",
    # software design / generic
    "implementation", "implementations", "concepts", "Concepts",
    "framework", "frameworks", "Frameworks", "context", "Context", "engineering", "Engineering",
    "programming", "Programming", "scraping", "Scraping",
    "design", "Design", "interface", "Interface", "micro", "Micro",
    "workflow", "Workflow", "testing", "Testing", "software", "Software",
    "infra", "Infra", "adoption", "Adoption",
    "improve", "Improve", "develops", "Develops", "helps", "Helps",
    "works", "Works", "drive", "Drive",
    "code", "Code", "documentation", "Documentation",
    "meet", "Meet", "within", "Within", "etc", "Etc",
    "workflows", "Workflows", "strategies", "Strategies",
    "prompts", "Prompts", "back-end", "EngOps", "engops",
    "needs", "Needs", "accuracy", "relevance", "consistency",
    "Problem-solving", "problem-solving",
}

ALL_STOP: set[str] = _NLTK_STOP | _DOMAIN_NOISE


# ── public entry point ────────────────────────────────────────────────────────

def extract_jd_keywords_v2(job_description: str) -> dict:
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

    required_keywords = _filter_tokens(req_lines)
    preferred_keywords = _filter_tokens(pref_lines)

    # Fallback: no section headings found — filter all lines
    if not required_keywords and not preferred_keywords:
        required_keywords = _filter_tokens(lines)

    req_set = {k.casefold() for k in required_keywords}
    preferred_keywords = [k for k in preferred_keywords if k.casefold() not in req_set]

    keywords = dedupe_preserve_order(required_keywords + preferred_keywords)
    boolean_query = _build_boolean_query(required_keywords, preferred_keywords)

    return {
        "keywords": keywords,
        "required_keywords": required_keywords,
        "preferred_keywords": preferred_keywords,
        "boolean_query": boolean_query,
    }


# ── two-pass token filter ─────────────────────────────────────────────────────

def _filter_tokens(lines: list[str]) -> list[str]:
    """
    Pass 1 — multi-word phrase matching from skills DB (longest first).
             Tracks character spans so matched words aren't re-tokenized.
    Pass 2 — single-word tokenization on remaining text.
             Stop-word filtered — non-English, non-noise tokens kept.
    """
    text = _PARENS_RE.sub(" ", " ".join(lines))
    text = re.sub(r"(\w)\s*/\s*(\w)", r"\1 \2", text)
    text_lower = text.lower()

    matched: list[str] = []
    consumed: list[tuple[int, int]] = []   # char spans already claimed by a phrase

    # ── Pass 1: multi-word DB phrases ────────────────────────────────────────
    for skill in _SKILLS_FLAT:
        if len(skill.split()) < 2:
            continue  # single words handled in pass 2
        needle = skill.lower()
        pattern = rf"(?<![a-z]){re.escape(needle)}(?![a-z])"
        for m in re.finditer(pattern, text_lower):
            span = (m.start(), m.end())
            # Skip if this span overlaps an already-consumed span
            if any(c[0] <= span[0] < c[1] or span[0] <= c[0] < span[1] for c in consumed):
                continue
            matched.append(skill)
            consumed.append(span)

    # ── Pass 2: single-word stop-word filter on unconsumed text ──────────────
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9\+\-\.#]*", text):
        token = m.group().rstrip(".")
        if not token:
            continue
        start, end = m.start(), m.start() + len(token)

        # Skip tokens that fall inside a phrase already matched in pass 1
        if any(c[0] <= start and end <= c[1] for c in consumed):
            continue
        if len(token) < 2:
            continue
        if token.lower() in ALL_STOP:
            continue
        matched.append(token)

    return dedupe_preserve_order(matched)


# ── boolean query ─────────────────────────────────────────────────────────────

def _build_boolean_query(required: list[str], preferred: list[str]) -> str:
    parts: list[str] = []
    if required:
        parts.append(f"({' OR '.join(required)})")
    if preferred:
        parts.append(f"({' OR '.join(preferred)})")
    return " AND ".join(parts)
