from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz

from app.parser.section_grouper import ResumeSectionGrouper


MIN_SELECTABLE_CHARS = 100
MIN_ALPHA_RATIO = 0.35


@dataclass(slots=True)
class PDFLine:
    page: int
    text: str
    bbox: list[float]
    block_no: int
    line_no: int


@dataclass(slots=True)
class PDFQuality:
    selectable: bool
    reason: str
    total_chars: int
    alpha_chars: int
    alpha_ratio: float
    line_count: int


class PDFResumeExtractor:
    """Extract structured line data from selectable-text PDF resumes."""

    def __init__(self) -> None:
        self.section_grouper = ResumeSectionGrouper()

    def extract(self, pdf_path: str | Path) -> dict[str, Any]:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

        lines = self.extract_lines(path)
        grouped_sections = self.section_grouper.group(lines)
        return grouped_sections

    def extract_lines(self, pdf_path: str | Path) -> list[PDFLine]:
        document = fitz.open(pdf_path)
        extracted: list[PDFLine] = []

        try:
            for page_index, page in enumerate(document):
                page_lines = self._extract_page_lines(page, page_index + 1)
                extracted.extend(page_lines)
        finally:
            document.close()

        return extracted

    def assess_quality(self, lines: list[PDFLine]) -> PDFQuality:
        text = self.lines_to_text(lines)
        total_chars = len(text)
        alpha_chars = sum(char.isalpha() for char in text)
        alpha_ratio = (alpha_chars / total_chars) if total_chars else 0.0

        if not text.strip():
            return PDFQuality(
                selectable=False,
                reason="no_text_found",
                total_chars=0,
                alpha_chars=0,
                alpha_ratio=0.0,
                line_count=len(lines),
            )

        if total_chars < MIN_SELECTABLE_CHARS:
            return PDFQuality(
                selectable=False,
                reason="too_little_text",
                total_chars=total_chars,
                alpha_chars=alpha_chars,
                alpha_ratio=alpha_ratio,
                line_count=len(lines),
            )

        if alpha_ratio < MIN_ALPHA_RATIO:
            return PDFQuality(
                selectable=False,
                reason="low_text_quality",
                total_chars=total_chars,
                alpha_chars=alpha_chars,
                alpha_ratio=alpha_ratio,
                line_count=len(lines),
            )

        return PDFQuality(
            selectable=True,
            reason="ok",
            total_chars=total_chars,
            alpha_chars=alpha_chars,
            alpha_ratio=alpha_ratio,
            line_count=len(lines),
        )

    def lines_to_text(self, lines: list[PDFLine]) -> str:
        return "\n".join(line.text for line in lines)

    def dump_json(self, payload: dict[str, Any], output_path: str | Path) -> None:
        path = Path(output_path)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    def _page_count(self, pdf_path: str | Path) -> int:
        document = fitz.open(pdf_path)
        try:
            return len(document)
        finally:
            document.close()

    def _extract_page_lines(self, page: fitz.Page, page_number: int) -> list[PDFLine]:
        text_dict = page.get_text("dict")
        lines: list[PDFLine] = []

        blocks = sorted(
            [block for block in text_dict.get("blocks", []) if block.get("type") == 0],
            key=lambda block: (round(block["bbox"][1], 1), round(block["bbox"][0], 1)),
        )

        for block_index, block in enumerate(blocks):
            block_lines = block.get("lines", [])
            sorted_lines = sorted(
                block_lines,
                key=lambda line: (
                    round(line["bbox"][1], 1),
                    round(line["bbox"][0], 1),
                ),
            )

            for line_index, line in enumerate(sorted_lines):
                spans = line.get("spans", [])
                if not spans:
                    continue

                line_text = self._normalize_line("".join(span.get("text", "") for span in spans))
                if not line_text:
                    continue

                x0 = min(span["bbox"][0] for span in spans)
                y0 = min(span["bbox"][1] for span in spans)
                x1 = max(span["bbox"][2] for span in spans)
                y1 = max(span["bbox"][3] for span in spans)

                lines.append(
                    PDFLine(
                        page=page_number,
                        text=line_text,
                        bbox=[round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
                        block_no=block_index,
                        line_no=line_index,
                    )
                )

        return lines

    def _normalize_line(self, text: str) -> str:
        replacements = {
            "\u2022": "-",
            "\u25cf": "-",
            "\u00a0": " ",
            "\u2013": "-",
            "\u2014": "-",
            "\u2212": "-",
            "\ufb01": "fi",
            "\ufb02": "fl",
            "\ufb03": "ffi",
            "\ufb04": "ffl",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()
