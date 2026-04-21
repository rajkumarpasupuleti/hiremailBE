from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_API_KEY_ENV = "GEMINI_API_KEY"


class GeminiResumeFormatter:
    """Format grouped resume sections into structured JSON using Gemini."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        api_key_env: str = DEFAULT_API_KEY_ENV,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.api_key_env = api_key_env

    def format(self, grouped_resume: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError(
                f"Missing Gemini API key. Set it in the {self.api_key_env} environment variable."
            )

        prompt = self._build_prompt(grouped_resume)
        response_payload = self._generate_content(prompt)
        response_text = self._extract_response_text(response_payload)
        cleaned_response = self._strip_code_fences(response_text)

        try:
            parsed = json.loads(cleaned_response)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Gemini response was not valid JSON. "
                f"Response text: {cleaned_response}"
            ) from exc

        return parsed

    def _build_prompt(self, grouped_resume: dict[str, Any]) -> str:
        grouped_json = json.dumps(grouped_resume, ensure_ascii=True, indent=2)
        return (
            "You are a resume parser.\n"
            "Convert the grouped resume JSON below into a clean structured resume JSON.\n"
            "Output JSON only. No markdown. No explanation.\n"
            "Do not invent facts.\n"
            "Normalize obvious capitalization issues when safe.\n"
            "Merge broken description lines into coherent sentences.\n"
            "If a field is unknown, use an empty string, empty array, or null.\n"
            "Use this schema exactly:\n"
            "{\n"
            '  "person_name": "",\n'
            '  "profile": "",\n'
            '  "summary": "",\n'
            '  "email": "",\n'
            '  "phone": "",\n'
            '  "address": "",\n'
            '  "skills": [],\n'
            '  "education": [\n'
            "    {\n"
            '      "institution": null,\n'
            '      "degree": "",\n'
            '      "level": null,\n'
            '      "start_date": "",\n'
            '      "end_date": "",\n'
            '      "location": "",\n'
            '      "description": null\n'
            "    }\n"
            "  ],\n"
            '  "work_experience": [\n'
            "    {\n"
            '      "title": "",\n'
            '      "company": "",\n'
            '      "start_date": "",\n'
            '      "end_date": null,\n'
            '      "location": "",\n'
            '      "description": ""\n'
            "    }\n"
            "  ],\n"
            '  "personal_projects": [\n'
            "    {\n"
            '      "name": "",\n'
            '      "date": "",\n'
            '      "description": ""\n'
            "    }\n"
            "  ],\n"
            '  "certificates": [\n'
            "    {\n"
            '      "name": "",\n'
            '      "date": "",\n'
            '      "description": ""\n'
            "    }\n"
            "  ],\n"
            '  "languages": [\n'
            "    {\n"
            '      "language": "",\n'
            '      "proficiency": ""\n'
            "    }\n"
            "  ],\n"
            '  "interests": []\n'
            "}\n\n"
            "Grouped resume JSON:\n"
            f"{grouped_json}\n"
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
            raise RuntimeError(
                f"Gemini request failed with HTTP {exc.code}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Gemini request failed: {exc.reason}") from exc

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
