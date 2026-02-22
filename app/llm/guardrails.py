"""
llm/guardrails.py — Three-layer safety and grounding controls.

Layer 1 (pre-call):  Input validation before any LLM call
Layer 2 (in-prompt): Constraint block in system prompt (handled by prompt_builder.py)
Layer 3 (post-call): Lightweight grounding check on the response

Why guardrails matter for this use case:
- A resume assistant that invents skills or experience is actively harmful
- Candidates might rely on the gap analysis to make career decisions
- Hallucinated skills in a gap analysis could lead to wasted interview prep

Trade-off acknowledged: The post-call grounding verifier adds a second LLM
call (~100ms on 8GB / llama3.2:3b). In production I would replace this with
OpenAI's JSON mode with source citations, eliminating the second call entirely.
For the 3B local model, this verification step meaningfully reduces hallucination.
"""
import httpx
from dataclasses import dataclass
from app.config import get_settings
from app.retrieval.vector_store import get_indexed_docs
from app.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GuardResult:
    passed: bool
    reason: str  # Human-readable explanation if check fails


# ── Layer 1: Pre-call Input Guardrails ────────────────────────────────────────

def validate_pre_call(
    query: str,
    intent: str,
    jd_id: str | None,
) -> GuardResult:
    """
    Validate inputs before making any LLM call.
    Returns GuardResult(passed=False, reason=...) if something is wrong.
    """
    indexed = get_indexed_docs()

    # Check 1: Resume required for resume_lookup and gap_analysis
    if intent in ("resume_lookup", "gap_analysis", "fit_score", "interview_prep"):
        if not indexed["has_resume"]:
            return GuardResult(
                passed=False,
                reason="No resume has been uploaded yet. Please upload your resume using the sidebar before asking questions about your profile or fit.",
            )

    # Check 2: At least one JD required for JD-dependent intents
    if intent in ("jd_lookup", "gap_analysis", "fit_score", "interview_prep"):
        if not indexed["job_ids"]:
            return GuardResult(
                passed=False,
                reason="No job descriptions have been uploaded yet. Please upload at least one job description before asking about job fit or gaps.",
            )

    # Check 3: Specific jd_id must exist if referenced
    if jd_id and jd_id not in indexed["job_ids"]:
        available = ", ".join(indexed["job_ids"])
        return GuardResult(
            passed=False,
            reason=f"Job '{jd_id}' not found. Available job IDs: {available}. Please reference one of these in your query.",
        )

    # Check 4: Basic query sanity
    if len(query.strip()) < 3:
        return GuardResult(passed=False, reason="Query is too short. Please ask a complete question.")

    return GuardResult(passed=True, reason="OK")


# ── Layer 3: Post-call Grounding Verifier ─────────────────────────────────────

GROUNDING_PROMPT = """You are a fact-checking assistant. Read the CONTEXT and the RESPONSE below.

CONTEXT (the only facts the response should reference):
{context}

RESPONSE to check:
{response}

Does the response make any specific claims about skills, job titles, companies, dates, or experience that are NOT present in the CONTEXT above?

Answer with ONLY one word: YES or NO

If the response says "not enough information" or declines to answer, that is fine — answer NO."""


def check_grounding(response: str, resume_chunks: list[dict], jd_chunks: list[dict]) -> GuardResult:
    """
    Post-call grounding check.

    Verifies the LLM's response doesn't contain claims not supported by
    the retrieved context. Returns GuardResult indicating pass/fail.

    This is a best-effort check — the verifier LLM can also hallucinate,
    but the combination of constrained prompt + independent verification
    meaningfully reduces grounding failures.
    """
    # Skip check if response is short (likely a "not enough info" response)
    if len(response.strip()) < 50:
        return GuardResult(passed=True, reason="Response too short to verify — skipping")

    # Build compact context for the verifier (first 800 chars of each chunk)
    context_parts = []
    for chunk in resume_chunks[:3]:
        context_parts.append(chunk["text"][:400])
    for chunk in jd_chunks[:3]:
        context_parts.append(chunk["text"][:400])

    if not context_parts:
        return GuardResult(passed=True, reason="No context to verify against")

    context = "\n---\n".join(context_parts)
    prompt = GROUNDING_PROMPT.format(context=context, response=response[:1000])

    settings = get_settings()
    try:
        url = f"{settings.ollama_base_url}/api/chat"
        payload = {
            "model": settings.ollama_llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 5},
        }
        with httpx.Client(timeout=60.0) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            verdict = r.json()["message"]["content"].strip().upper()

        passed = "NO" in verdict  # NO hallucination detected = passed

        logger.info(
            "grounding_check",
            extra={"verdict": verdict, "passed": passed},
        )
        return GuardResult(passed=passed, reason=f"Grounding verifier: {verdict}")

    except Exception as e:
        # If the verifier itself fails, don't block the response
        logger.warning("grounding_check_error", extra={"error": str(e)})
        return GuardResult(passed=True, reason="Verifier unavailable — check skipped")


def add_grounding_caveat(response: str) -> str:
    """
    Add a soft disclaimer when grounding check flags a potential issue.
    We don't suppress the response — we add a caveat so the user can judge.
    """
    caveat = (
        "\n\n⚠️ *Note: Some claims in this response may not be directly supported by "
        "the uploaded documents. Please verify against your original resume and job description.*"
    )
    return response + caveat
