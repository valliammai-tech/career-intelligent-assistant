"""
llm/prompt_builder.py — Composable system prompt assembly.

The system prompt is NOT static. It's assembled from three blocks per request:
  1. PERSONA BLOCK   — static, defines role and tone
  2. CONTEXT BLOCK   — dynamic, injected retrieved chunks + skills manifests
  3. CONSTRAINT BLOCK — static, grounding rules the LLM must follow

Why three blocks?
- PERSONA is always the same — cache it
- CONTEXT changes every request — build it from retrieval results
- CONSTRAINT is always the same — but must come AFTER context so the
  model applies rules to the specific evidence it was just given

The intent param shapes the response format instruction:
  gap_analysis -> structured PRESENT/PARTIAL/MISSING format
  fit_score    -> numerical score with justification
  interview_prep -> bullet list of likely questions
  resume/jd lookup -> concise factual answer
"""
import json
from app.retrieval.vector_store import get_indexed_docs

PERSONA_BLOCK = """You are a Career Intelligence Analyst. Your role is to help candidates understand their fit for roles, identify skill gaps, and prepare for interviews.

You reason from evidence — you cite only information present in the provided documents. You are precise, constructive, and direct."""

CONSTRAINT_BLOCK = """RULES YOU MUST FOLLOW:
1. Only cite skills and experience that appear in the RESUME CONTEXT or JD CONTEXT above.
2. Never invent job titles, companies, dates, or skills not present in the documents.
3. If you cannot determine something from the context, say "Not enough information in the provided documents."
4. For gap analysis, use the format: ✅ PRESENT | ⚠️ PARTIAL | ❌ MISSING — with one line of evidence.
5. Keep answers focused and structured. Avoid generic career advice not grounded in the documents."""

FORMAT_INSTRUCTIONS = {
    "resume_lookup":  "Answer concisely based on the resume context. Cite the specific section.",
    "jd_lookup":      "Answer concisely based on the JD context. Cite the specific requirement.",
    "gap_analysis":   "Structure your answer as a skill-by-skill gap analysis using ✅ PRESENT | ⚠️ PARTIAL | ❌ MISSING. End with a 2-sentence overall assessment.",
    "fit_score":      "Give an overall fit score from 0-100 with a one-paragraph justification based on skill overlap. Then list the top 3 strengths and top 3 gaps.",
    "interview_prep": "List 5-7 likely interview questions for this role based on the JD requirements and the candidate's specific background. For each question, add a one-line tip based on the resume.",
}


def build_messages(
    query: str,
    intent: str,
    resume_chunks: list[dict],
    jd_chunks: list[dict],
    jd_id: str | None,
    conversation_history: list[dict],
    gap_report=None,
    ranking=None,
) -> list[dict]:
    """
    Assemble the full messages list for the LLM call.

    Returns:
        [{"role": "system", "content": "..."}, {"role": "user", ...}, ...]
    """
    system_content = _build_system_prompt(
        intent=intent,
        resume_chunks=resume_chunks,
        jd_chunks=jd_chunks,
        jd_id=jd_id,
        gap_report=gap_report,
        ranking=ranking,
    )

    messages: list[dict] = [{"role": "system", "content": system_content}]

    # Inject conversation history (sliding window — managed by memory.py)
    messages.extend(conversation_history)

    # Current user query
    messages.append({"role": "user", "content": query})

    return messages


def _build_system_prompt(
    intent: str,
    resume_chunks: list[dict],
    jd_chunks: list[dict],
    jd_id: str | None,
    gap_report=None,
    ranking=None,
) -> str:
    """Assemble system prompt from the three blocks."""
    from app.retrieval.gap_analyser import format_gap_report_for_prompt, format_ranking_for_prompt
    parts = [PERSONA_BLOCK, ""]

    # ── CONTEXT BLOCK ─────────────────────────────────────────────────────────
    if resume_chunks:
        resume_text = _format_chunks(resume_chunks)
        parts.append(f"--- RESUME CONTEXT ---\n{resume_text}")
        parts.append("")

        # Inject skills manifest from the first chunk's metadata
        manifest_raw = resume_chunks[0]["metadata"].get("skills_manifest", "{}")
        try:
            manifest = json.loads(manifest_raw)
            parts.append(f"CANDIDATE SKILLS MANIFEST:\n{_format_manifest(manifest)}")
            parts.append("")
        except Exception:
            pass

    if jd_chunks:
        jd_label = f"JD CONTEXT ({jd_id})" if jd_id else "JD CONTEXT"
        jd_text = _format_chunks(jd_chunks)
        parts.append(f"--- {jd_label} ---\n{jd_text}")
        parts.append("")

        # Inject JD skills manifest
        manifest_raw = jd_chunks[0]["metadata"].get("skills_manifest", "{}")
        try:
            manifest = json.loads(manifest_raw)
            parts.append(f"JD REQUIRED SKILLS:\n{_format_manifest(manifest)}")
            parts.append("")
        except Exception:
            pass

    # Inject pre-computed gap analysis if available (gap_analysis intent)
    if gap_report is not None:
        parts.append(format_gap_report_for_prompt(gap_report))
        parts.append("")

    # Inject multi-JD ranking if available (fit_score intent, no specific jd_id)
    if ranking is not None:
        parts.append(format_ranking_for_prompt(ranking))
        parts.append("")

    # Warn if no documents are available
    if not resume_chunks and not jd_chunks:
        indexed = get_indexed_docs()
        available_jds = ", ".join(indexed["job_ids"]) or "none"
        parts.append(
            f"WARNING: No relevant documents retrieved for this query.\n"
            f"Resume indexed: {indexed['has_resume']}\n"
            f"Available JD IDs: {available_jds}"
        )
        parts.append("")

    # ── FORMAT INSTRUCTION ────────────────────────────────────────────────────
    format_instruction = FORMAT_INSTRUCTIONS.get(
        intent,
        FORMAT_INSTRUCTIONS["gap_analysis"]
    )
    parts.append(f"RESPONSE FORMAT: {format_instruction}")
    parts.append("")

    # ── CONSTRAINT BLOCK ──────────────────────────────────────────────────────
    parts.append(CONSTRAINT_BLOCK)

    return "\n".join(parts)


def _format_chunks(chunks: list[dict]) -> str:
    """
    Format retrieved chunks for prompt injection.
    Includes section label and page number for traceability.
    """
    formatted = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        section = meta.get("section", "Unknown")
        page = meta.get("page", "?")
        text = chunk.get("text", "").strip()
        formatted.append(f"[{i}] Section: {section} | Page: {page}\n{text}")
    return "\n\n".join(formatted)


def _format_manifest(manifest: dict) -> str:
    """Compact human-readable skills manifest for prompt injection."""
    lines = []
    if manifest.get("hard_skills"):
        lines.append(f"Hard skills: {', '.join(manifest['hard_skills'])}")
    if manifest.get("soft_skills"):
        lines.append(f"Soft skills: {', '.join(manifest['soft_skills'])}")
    if manifest.get("tools_and_technologies"):
        lines.append(f"Tools: {', '.join(manifest['tools_and_technologies'])}")
    if manifest.get("experience_years"):
        lines.append(f"Experience: {manifest['experience_years']} years")
    if manifest.get("seniority_level"):
        lines.append(f"Seniority: {manifest['seniority_level']}")
    return "\n".join(lines) if lines else "No skills manifest available"
