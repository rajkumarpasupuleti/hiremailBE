"""
Utilities for cleaning and normalizing text from parser payloads.
"""

from __future__ import annotations

import re
import unicodedata

MOJIBAKE_MARKERS = ("Ã", "â", "Â", "ð", "€™", "â€™", "â€œ", "â€", "\ufffd")

SMART_CHAR_REPLACEMENTS = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2022": "- ",
    "\u00a0": " ",
}

KNOWN_ARTIFACT_REPLACEMENTS = {
    "\u00c3\u00a2\u00e2?\u00ac\u00e2?\u00a2": "'",
    "\u00c3\u00a2\u00e2?\u00ac\u00e2??": " - ",
}


def _score_mojibake(text: str) -> int:
    """Lower score means the candidate text looks cleaner."""
    return sum(text.count(marker) for marker in MOJIBAKE_MARKERS)


def _maybe_fix_mojibake(text: str) -> str:
    """
    Try common mojibake repairs such as:
    "BachelorÃ¢â‚¬â„¢s" -> "Bachelor's"
    """
    if not any(marker in text for marker in MOJIBAKE_MARKERS):
        return text

    best = text

    for _ in range(3):
        candidates = [best]
        for source_encoding in ("latin1", "cp1252"):
            try:
                candidates.append(best.encode(source_encoding).decode("utf-8"))
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue

        updated = min(candidates, key=_score_mojibake)
        if updated == best:
            break
        best = updated

    return best


def clean_text(value: object) -> str:
    """Normalize unicode, whitespace, and common encoding artifacts."""
    if value is None:
        return ""

    text = str(value)
    text = _maybe_fix_mojibake(text)

    for source, target in KNOWN_ARTIFACT_REPLACEMENTS.items():
        text = text.replace(source, target)

    for source, target in SMART_CHAR_REPLACEMENTS.items():
        text = text.replace(source, target)

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\ufffd", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def dedupe_preserve_order(values: list[str]) -> list[str]:
    """Remove empty and duplicate values while preserving order."""
    seen = set()
    result = []

    for value in values:
        cleaned = clean_text(value)
        if not cleaned:
            continue

        key = cleaned.casefold()
        if key in seen:
            continue

        seen.add(key)
        result.append(cleaned)

    return result
