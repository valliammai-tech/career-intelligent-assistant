"""
ingestion/skill_extractor.py — Structured skill extraction via LLM.

Called once per document at ingest time (not at query time).
Produces a compact JSON skills manifest stored alongside chunks.

Why extract at ingest, not at query time?
- Skill manifests are static — a resume's skills don't change between queries
- Storing them means gap analysis queries don't need an extra LLM call
- The manifest is injected into the prompt context for every gap analysis query

The LLM is asked to return strict JSON. We use a tight prompt with explicit
examples to minimise format variation. If parsing fails, we return an empty
manifest rather than crashing — the system degrades gracefully.
"""
import json
import httpx
from app.config import get_settings
from app.observability.logger import get_logger

logger = get_logger(__name__)

EMPTY_MANIFEST: dict = {
    "hard_skills": [],
    "soft_skills": [],
    "tools_and_technologies": [],
    "experience_years": None,
    "seniority_level": None,
}

EXTRACTION_PROMPT = """You are a skills extraction assistant. Read the document below and extract structured information.

Return ONLY a valid JSON object with exactly these fields (no markdown, no explanation, just JSON):
{{
  "hard_skills": ["Python", "FastAPI", "SQL"],
  "soft_skills": ["Team leadership", "Communication"],
  "tools_and_technologies": ["Docker", "AWS", "Git"],
  "experience_years": 5,
  "seniority_level": "senior"
}}

Rules:
- hard_skills: specific technical skills and domain knowledge
- soft_skills: interpersonal and leadership skills
- tools_and_technologies: specific tools, platforms, frameworks, cloud services
- experience_years: total years of relevant experience as a number, or null if not mentioned
- seniority_level: one of "junior", "mid", "senior", "lead", "principal", or null

Document:
---
{document_text}
---

JSON only:"""


def extract_skills(text: str) -> dict:
    """
    Extract a structured skills manifest from document text.

    Returns a dict matching EMPTY_MANIFEST schema.
    Never raises — returns EMPTY_MANIFEST on any failure.
    """
    settings = get_settings()

    # Truncate to ~2000 chars to stay within context limits for 3B model
    # The key skills are usually in the first portion of any document
    truncated = text[:3000] if len(text) > 3000 else text

    prompt = EXTRACTION_PROMPT.format(document_text=truncated)

    try:
        raw = _call_llm(prompt, settings)
        manifest = _parse_json_response(raw)
        logger.info("skills_extracted", extra={"skills_count": len(manifest.get("hard_skills", []))})
        return manifest
    except Exception as e:
        logger.warning("skill_extraction_failed", extra={"error": str(e)})
        return EMPTY_MANIFEST.copy()


def _call_llm(prompt: str, settings) -> str:
    """Call Ollama chat endpoint and return the raw text response."""
    url = f"{settings.ollama_base_url}/api/chat"
    payload = {
        "model": settings.ollama_llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.0,   # Deterministic for structured extraction
            "num_predict": 300,   # Skills manifest is short — cap tokens
        },
    }
    with httpx.Client(timeout=120.0) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]


def _parse_json_response(raw: str) -> dict:
    """
    Extract and parse JSON from LLM response.

    LLMs sometimes wrap JSON in markdown fences even when told not to.
    We strip those defensively before parsing.
    """
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        text = "\n".join(lines[1:-1]).strip()

    # Find the JSON object boundaries
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in LLM response")

    parsed = json.loads(text[start:end])

    # Validate expected keys are present
    for key in EMPTY_MANIFEST:
        if key not in parsed:
            parsed[key] = EMPTY_MANIFEST[key]

    return parsed
