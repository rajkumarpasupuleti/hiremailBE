"""
JD keyword extraction V3 — skillNer (31k EMSI skills) + spaCy POS fallback.

Two-pass hybrid:
  Pass 1 — skillNer positive matching against 31k EMSI skills dataset.
            Returns exact text spans for known tech/soft skills.
            Covers multi-word phrases ("machine learning", "spring boot") natively.

  Pass 2 — spaCy POS single-token sweep on what skillNer missed.
            Keeps NOUN/PROPN tokens that are not stop words or domain noise.
            Catches newer terms not yet in skillNer's dataset (LangChain, RAG, etc.)

Section detection and inline-preferred promotion reused from v1.
"""

from __future__ import annotations

import re
import logging

import spacy
from spacy.matcher import PhraseMatcher
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor

from app.engine.text_utils import clean_text, dedupe_preserve_order
from app.services.jd_keywords import _NUMBERED_LIST_RE, _PARENS_RE, _extract_sections

logger = logging.getLogger(__name__)

# ── lazy singletons — load once on first request ──────────────────────────────

_nlp: spacy.language.Language | None = None
_skill_extractor: SkillExtractor | None = None


def _get_models() -> tuple[spacy.language.Language, SkillExtractor]:
    global _nlp, _skill_extractor
    if _nlp is None:
        logger.info("Loading spaCy en_core_web_sm + skillNer (first request only)...")
        _nlp = spacy.load("en_core_web_sm")
        _skill_extractor = SkillExtractor(_nlp, SKILL_DB, PhraseMatcher)
        logger.info("skillNer ready — 31k EMSI skills loaded.")
    return _nlp, _skill_extractor


# ── domain noise lemmas (spaCy pass only) ────────────────────────────────────
# Base-form lemmas for nouns that are never skills in a JD.
# Only needed for pass 2 (spaCy POS sweep) — skillNer handles known skills
# via positive dataset matching so these don't affect pass 1.

_DOMAIN_NOISE_LEMMAS: frozenset[str] = frozenset({
    # experience / time
    "experience", "year", "month",
    # abstract descriptors
    "knowledge", "skill", "ability", "familiarity",
    "proficiency", "exposure", "understanding", "expertise", "communication",
    # JD admin
    "requirement", "qualification", "responsibility", "certification",
    # education
    "degree", "bachelor", "master", "equivalent",
    # org / context
    "field", "area", "domain", "team", "member", "staff",
    "role", "position", "job", "title", "candidate",
    "company", "organization", "business", "environment",
    # generic tech nouns
    "system", "application", "platform", "tool",
    "technology", "solution", "service", "product", "database",
    # JD process
    "setting", "practice", "document", "contract",
    "calendar", "record", "activity", "task", "project",
    "procedure", "process", "operation",
    # design / generic
    "concept", "context", "framework", "workflow", "strategy",
    "software", "engineering", "programming", "testing", "design",
    "development", "implementation", "adoption", "code", "documentation",
    "building", "web",
    # job roles
    "developer", "engineer", "specialist", "analyst", "manager",
    "architect", "coordinator", "tester", "lead",
    # logistics
    "delivery", "deadline", "range", "hour", "day", "week", "stakeholder",
    # context nouns that slip through as NOUN/PROPN
    "matter", "subject", "volume", "transaction",
    # adjectives spaCy occasionally mis-tags as PROPN with small model
    "good", "great", "strong", "senior", "junior",
    # explicit plurals where PROPN lemma doesn't reduce (spaCy quirk)
    "systems", "technologies", "databases", "services", "solutions",
    "applications", "platforms", "tools", "frameworks",
    # universal context nouns — actions/logistics/prose that are never skills
    "call", "calls", "review", "reviews", "support", "work", "home",
    "location", "training", "schedule", "shift", "date",
    "download", "upload", "connection", "internet", "workspace",
    "compliance", "patient", "medication", "allergy", "effect",
    "conference", "highlight", "keyword", "keywords",
    "minimum", "maximum", "access", "ability", "use",
    # work arrangement / logistics
    "remote", "wfh", "outbound", "inbound", "onsite", "hybrid",
    "comfort", "mbps", "ct", "am", "pm",
    # verbs spaCy occasionally mis-tags as NOUN/PROPN
    "coordinate", "coordinating", "engaging", "seeking", "hiring",
})

# spaCy NER labels that are never skills — dates, numbers, locations, money
_NON_SKILL_NER_LABELS: frozenset[str] = frozenset({
    "DATE", "TIME", "CARDINAL", "ORDINAL", "MONEY", "PERCENT",
    "QUANTITY", "GPE", "LOC", "FAC",
})

_INLINE_PREFERRED_RE = re.compile(
    r"\b(preferred|is a plus|nice to have|bonus)\b", re.IGNORECASE
)

# skillNer score threshold — 1 = full/exact match only (no partial ngram noise)
_SKILL_SCORE_THRESHOLD = 1


# ── public entry point ────────────────────────────────────────────────────────

def extract_jd_keywords_v3(job_description: str) -> dict:
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

    required_keywords = _extract_keywords(req_lines)
    preferred_keywords = _extract_keywords(pref_lines)

    if not required_keywords and not preferred_keywords:
        required_keywords = _extract_keywords(lines)

    required_keywords, inline_preferred = _split_inline_preferred(required_keywords, req_lines)
    preferred_keywords = dedupe_preserve_order(inline_preferred + preferred_keywords)

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


# ── extraction core ───────────────────────────────────────────────────────────

def _extract_keywords(lines: list[str]) -> list[str]:
    if not lines:
        return []

    nlp, extractor = _get_models()
    matched: list[str] = []
    consumed_lower: set[str] = set()

    for line in lines:
        # Split on "/" and "," — keeps tech lists as separate segments
        # "Kafka/MQ/Rabbit" → ["Kafka", "MQ", "Rabbit"]
        segments = re.split(r"\s*/\s*|\s*,\s*", line)
        for segment in segments:
            segment = _PARENS_RE.sub(" ", segment).strip()
            segment = re.sub(r"^\s*and\s+", "", segment, flags=re.IGNORECASE).strip()
            if not segment:
                continue

            # ── Pass 1: skillNer — positive match against 31k EMSI dataset ──────
            try:
                result = extractor.annotate(segment)

                # full_matches: exact multi-word skill matches (highest confidence)
                for m in result["results"]["full_matches"]:
                    val = _restore_case(m["doc_node_value"], segment)
                    key = val.lower()
                    if key not in consumed_lower:
                        matched.append(val)
                        consumed_lower.add(key)
                        # Mark each component word consumed so spaCy pass
                        # doesn't re-extract "Computer" + "Science" when
                        # skillNer already returned "Computer Science"
                        for word in key.split():
                            consumed_lower.add(word)

                # ngram_scored: single/partial matches — only take score == 1
                for m in result["results"]["ngram_scored"]:
                    if int(m["score"]) < _SKILL_SCORE_THRESHOLD:
                        continue
                    val = _restore_case(m["doc_node_value"], segment)
                    key = val.lower()
                    if key not in consumed_lower:
                        # Single-word skillNer results: run through same noise
                        # filter as spaCy pass to catch dataset false positives
                        # like "Good" from "Good Manufacturing Practice" phrases
                        if len(val.split()) == 1:
                            tok_doc = nlp(val)
                            tok = tok_doc[0]
                            if tok.is_stop or tok.lemma_.lower() in _DOMAIN_NOISE_LEMMAS:
                                continue
                        matched.append(val)
                        consumed_lower.add(key)
                        for word in key.split():
                            consumed_lower.add(word)

            except Exception:
                logger.exception("skillNer failed on segment=%r, continuing with POS pass", segment)

            # ── Pass 2: spaCy POS — catches newer terms not yet in skillNer ──────
            doc = nlp(segment)
            # Build set of token indices that belong to DATE/TIME/CARDINAL etc. entities
            ner_noise_indices: set[int] = {
                tok.i for ent in doc.ents
                if ent.label_ in _NON_SKILL_NER_LABELS
                for tok in ent
            }
            for tok in doc:
                if tok.i in ner_noise_indices:
                    continue
                if tok.pos_ not in ("NOUN", "PROPN"):
                    continue
                if tok.is_stop:
                    continue
                if tok.lemma_.lower() in _DOMAIN_NOISE_LEMMAS or tok.text.lower() in _DOMAIN_NOISE_LEMMAS:
                    continue
                token = tok.text.rstrip(".")
                if len(token) < 2:
                    continue
                key = token.lower()
                if key not in consumed_lower:
                    matched.append(token)
                    consumed_lower.add(key)

    return dedupe_preserve_order(matched)


# ── inline preferred splitter ─────────────────────────────────────────────────

def _split_inline_preferred(
    keywords: list[str],
    source_lines: list[str],
) -> tuple[list[str], list[str]]:
    nlp, _ = _get_models()
    preferred_set: set[str] = set()

    for line in source_lines:
        if not _INLINE_PREFERRED_RE.search(line):
            continue
        for segment in re.split(r"\s*/\s*|\s*,\s*", line):
            segment = _PARENS_RE.sub(" ", segment).strip()
            if not segment:
                continue
            doc = nlp(segment)
            for tok in doc:
                if (
                    tok.pos_ in ("NOUN", "PROPN")
                    and not tok.is_stop
                    and tok.lemma_.lower() not in _DOMAIN_NOISE_LEMMAS
                ):
                    preferred_set.add(tok.text.casefold())

    required = [kw for kw in keywords if kw.casefold() not in preferred_set]
    preferred = [kw for kw in keywords if kw.casefold() in preferred_set]
    return required, preferred


# ── helpers ───────────────────────────────────────────────────────────────────

def _restore_case(lowercased: str, source: str) -> str:
    """Find original casing of a skillNer lowercased match in the source segment."""
    m = re.search(re.escape(lowercased), source, re.IGNORECASE)
    return m.group(0) if m else lowercased.title()


def _build_boolean_query(required: list[str], preferred: list[str]) -> str:
    parts: list[str] = []
    if required:
        parts.append(f"({' OR '.join(required)})")
    if preferred:
        parts.append(f"({' OR '.join(preferred)})")
    return " AND ".join(parts)
