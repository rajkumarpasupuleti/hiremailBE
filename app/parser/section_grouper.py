from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.parser.pdf_extractor import PDFLine


SECTION_ALIASES = {
    "summary": "summary",
    "profile": "summary",
    "education": "education",
    "skills": "skills",
    "work experience": "work_experience",
    "experience": "work_experience",
    "personal projects": "personal_projects",
    "projects": "personal_projects",
    "certificates": "certificates",
    "certifications": "certificates",
    "languages": "languages",
    "interests": "interests",
}

SECTION_KEYS = [
    "summary",
    "education",
    "skills",
    "work_experience",
    "personal_projects",
    "certificates",
    "languages",
    "interests",
    "other",
]

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:(?:\+\d{1,3}[\s-]?)?(?:\(?\d{2,5}\)?[\s-]?)?\d[\d\s-]{7,}\d)")


class ResumeSectionGrouper:
    """Group extracted PDF lines under resume section headings."""

    def group(self, lines: list[PDFLine]) -> dict[str, object]:
        grouped: dict[str, object] = {key: [] for key in SECTION_KEYS}
        header_lines: list[str] = []
        if not lines:
            grouped["header"] = self._build_header([])
            return grouped

        pages = self._group_by_page(lines)
        for page_number in sorted(pages):
            page_lines = pages[page_number]
            first_heading_y = self._first_heading_y(page_lines)
            split_x = self._page_split_x(page_lines)

            for line in self._sorted_lines(
                [line for line in page_lines if first_heading_y is not None and line.bbox[1] < first_heading_y]
            ):
                header_lines.append(self._clean_text(line.text))

            remaining_lines = [
                line for line in page_lines if first_heading_y is None or line.bbox[1] >= first_heading_y
            ]
            columns = self._group_by_column(remaining_lines, split_x)

            for column_lines in columns.values():
                current_section: str | None = None

                for line in self._sorted_lines(column_lines):
                    section_key = self._section_key(line.text)
                    if section_key:
                        current_section = section_key
                        continue

                    target = current_section or "other"
                    section_lines = grouped[target]
                    if isinstance(section_lines, list):
                        section_lines.append(self._clean_text(line.text))

        grouped["header"] = self._build_header(header_lines)
        return grouped

    def _group_by_page(self, lines: list[PDFLine]) -> dict[int, list[PDFLine]]:
        pages: dict[int, list[PDFLine]] = defaultdict(list)
        for line in lines:
            pages[line.page].append(line)
        return pages

    def _group_by_column(
        self, lines: list[PDFLine], split_x: float
    ) -> dict[str, list[PDFLine]]:
        columns = {"left": [], "right": []}
        for line in lines:
            if line.bbox[0] < split_x:
                columns["left"].append(line)
            else:
                columns["right"].append(line)
        return columns

    def _sorted_lines(self, lines: list[PDFLine]) -> list[PDFLine]:
        return sorted(lines, key=lambda line: (line.bbox[1], line.bbox[0], line.page))

    def _first_heading_y(self, lines: list[PDFLine]) -> float | None:
        heading_ys = [line.bbox[1] for line in lines if self._section_key(line.text)]
        if not heading_ys:
            return None
        return min(heading_ys)

    def _page_split_x(self, lines: list[PDFLine]) -> float:
        min_x = min(line.bbox[0] for line in lines)
        max_x = max(line.bbox[2] for line in lines)
        return (min_x + max_x) / 2

    def _section_key(self, text: str) -> str | None:
        normalized = re.sub(r"[^a-z]+", " ", text.lower()).strip()
        if not normalized:
            return None
        return SECTION_ALIASES.get(normalized)

    def _clean_text(self, text: str) -> str:
        replacements = {
            "\ufb01": "fi",
            "\ufb02": "fl",
            "\ufb03": "ffi",
            "\ufb04": "ffl",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+,", ",", text)
        return text

    def _build_header(self, header_lines: list[str]) -> dict[str, object]:
        cleaned_lines = [line for line in header_lines if line]
        email = self._extract_first_match(cleaned_lines, EMAIL_RE)
        phone = self._extract_first_match(cleaned_lines, PHONE_RE)

        remaining_lines: list[str] = []
        address = ""

        for line in cleaned_lines:
            stripped_line = line
            stripped_line = EMAIL_RE.sub("", stripped_line).strip(" |,")
            stripped_line = PHONE_RE.sub("", stripped_line).strip(" |,")

            if stripped_line:
                remaining_lines.append(stripped_line)

        person_name = remaining_lines[0] if remaining_lines else ""
        profile = remaining_lines[1] if len(remaining_lines) > 1 else ""
        summary = remaining_lines[2] if len(remaining_lines) > 2 else ""

        if len(remaining_lines) > 3:
            address = remaining_lines[3]

        other = remaining_lines[4:] if len(remaining_lines) > 4 else []

        return {
            "person_name": person_name,
            "profile": profile,
            "summary": summary,
            "email": email,
            "phone": phone,
            "address": address,
            "other": other,
        }

    def _extract_first_match(self, lines: list[str], pattern: re.Pattern[str]) -> str:
        for line in lines:
            match = pattern.search(line)
            if match:
                return match.group(0).strip()
        return ""
