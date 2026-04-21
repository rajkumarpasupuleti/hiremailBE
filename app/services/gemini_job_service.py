"""
Gemini-backed job requirement generation service.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


class GeminiJobRequirementsService:
    """Generate structured job requirement JSON from lightweight job metadata."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_model

    def generate(self, job_input: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            logger.error("Gemini job generation failed: missing API key")
            raise ValueError("Missing Gemini API key. Set GEMINI_API_KEY in the API .env file.")

        logger.info(
            "Gemini job generation started: title=%s company=%s",
            job_input.get("title", ""),
            job_input.get("company", ""),
        )
        prompt = self._build_prompt(job_input)
        response_payload = self._generate_content(prompt)
        response_text = self._extract_response_text(response_payload)
        cleaned_response = self._strip_code_fences(response_text)

        try:
            parsed = json.loads(cleaned_response)
            logger.info(
                "Gemini job generation completed: title=%s required_skills=%s preferred_skills=%s keywords=%s",
                parsed.get("job_title", ""),
                len(parsed.get("required_skills", [])),
                len(parsed.get("preferred_skills", [])),
                len(parsed.get("keywords", [])),
            )
            return parsed
        except json.JSONDecodeError as exc:
            logger.exception("Gemini job generation returned invalid JSON")
            raise ValueError(
                "Gemini job response was not valid JSON. "
                f"Response text: {cleaned_response}"
            ) from exc

    def _build_prompt(self, job_input: dict[str, Any]) -> str:
        job_input_json = json.dumps(job_input, ensure_ascii=True, indent=2)
        return (
            "You generate structured ATS-ready job requirement JSON.\n"
            "Use the provided job metadata and infer realistic requirements carefully.\n"
            "Output JSON only. No markdown. No explanation.\n"
            "Do not invent company-specific facts that are not implied by the title or summary.\n"
            "If experience is unclear, use 0.\n"
            "If education is unclear, use an empty string.\n"
            "Keep skills and keywords concise and deduplicated.\n"
            "Use this schema exactly:\n"
            "{\n"
            '  "job_title": "",\n'
            '  "company_name": "",\n'
            '  "location": "",\n'
            '  "work_mode": "",\n'
            '  "employment_type": "",\n'
            '  "seniority": "",\n'
            '  "job_summary": "",\n'
            '  "responsibilities": [],\n'
            '  "required_skills": [],\n'
            '  "preferred_skills": [],\n'
            '  "keywords": [],\n'
            '  "experience_years_required": 0,\n'
            '  "education_required": ""\n'
            "}\n\n"
            "Job metadata JSON:\n"
            f"{job_input_json}\n"
        )

    def _generate_content(self, prompt: str) -> dict[str, Any]:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ]
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            logger.exception("Gemini job HTTP error: status=%s", exc.code)
            raise RuntimeError(
                f"Gemini job request failed with HTTP {exc.code}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            logger.exception("Gemini job network error")
            raise RuntimeError(f"Gemini job request failed: {exc.reason}") from exc

    def _extract_response_text(self, response_payload: dict[str, Any]) -> str:
        candidates = response_payload.get("candidates", [])
        if not candidates:
            raise ValueError(f"No candidates returned by Gemini: {response_payload}")

        parts = candidates[0].get("content", {}).get("parts", [])
        texts = [part.get("text", "") for part in parts if part.get("text")]
        if not texts:
            raise ValueError(f"No text content returned by Gemini: {response_payload}")

        return "\n".join(texts).strip()

    def _strip_code_fences(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
        return stripped


def generate_job_requirements(job_input: dict[str, Any]) -> dict[str, Any]:
    """Convenience wrapper for Gemini-generated ATS job requirements."""
    service = GeminiJobRequirementsService()
    return service.generate(job_input)
