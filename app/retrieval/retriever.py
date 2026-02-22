"""
retrieval/retriever.py — Query intent classification and retrieval routing.

Intent classification determines WHICH collections to query and HOW to
combine results. Five intent classes, each with a different retrieval strategy.

Why classify intent at all?
A naive approach queries both collections for every request and hopes the
LLM figures out what the user wants. This wastes token budget on irrelevant
context and degrades answer quality for focused queries.

Intent classification adds ~100ms (one small LLM call) but significantly
improves retrieval precision. On 8GB / 3B model, it's worth the cost.
"""
import re
import httpx
from app.config import get_settings
from app.ingestion.embedder import embed_query
from app.retrieval.vector_store import (
    query_resume, query_jobs, mmr_rerank, get_indexed_docs
)
from app.observability.logger import get_logger

logger = get_logger(__name__)

INTENT_CLASSES = {
    "resume_lookup":  "Questions about the candidate's own background, experience, skills.",
    "jd_lookup":      "Questions about what a specific job requires.",
    "gap_analysis":   "Comparing candidate to a job — missing skills, fit analysis.",
    "fit_score":      "Overall match score or ranking across multiple jobs.",
    "interview_prep": "Interview questions or preparation advice for a specific role.",
}

INTENT_PROMPT = """Classify this query into exactly one of these five categories:
resume_lookup, jd_lookup, gap_analysis, fit_score, interview_prep

Definitions:
- resume_lookup: about the candidate's background, skills, experience, education
- jd_lookup: about what a specific job requires or offers
- gap_analysis: comparing candidate to job, missing skills, skill gaps, alignment
- fit_score: overall match score, ranking jobs, which job is best fit
- interview_prep: preparing for interviews, expected questions, how to prepare

Query: "{query}"

Reply with ONLY the category name, nothing else:"""


def classify_intent(query: str) -> str:
    """
    Classify query intent using the LLM.
    Falls back to 'gap_analysis' on any failure — the most generally useful intent.
    """
    settings = get_settings()
    prompt = INTENT_PROMPT.format(query=query)

    try:
        url = f"{settings.ollama_base_url}/api/chat"
        payload = {
            "model": settings.ollama_llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 20},
        }
        with httpx.Client(timeout=30.0) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            raw = r.json()["message"]["content"].strip().lower()

        # Extract intent from response
        for intent in INTENT_CLASSES:
            if intent in raw:
                logger.info("intent_classified", extra={"intent": intent, "query": query[:60]})
                return intent

    except Exception as e:
        logger.warning("intent_classification_failed", extra={"error": str(e)})

    return "gap_analysis"  # Safe default


def extract_jd_id_from_query(query: str) -> str | None:
    """
    Heuristically extract a jd_id reference from the query.
    Handles patterns like: Job #1, Job 1, job_1, job #2, JD 3, role 2
    """
    patterns = [
        r"job\s*#?\s*(\d+)",
        r"jd\s*#?\s*(\d+)",
        r"role\s*#?\s*(\d+)",
        r"position\s*#?\s*(\d+)",
        r"job_(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return f"job_{match.group(1)}"
    return None


def retrieve_context(query: str, explicit_jd_id: str | None = None) -> dict:
    """
    Main retrieval entry point.

    1. Classify intent
    2. Embed the query
    3. Route to appropriate collection(s)
    4. MMR re-rank results
    5. Return structured context ready for prompt injection

    Returns:
        {
            "intent": str,
            "resume_chunks": [...],
            "jd_chunks": [...],
            "jd_id": str | None,
        }
    """
    settings = get_settings()
    intent = classify_intent(query)

    # Resolve jd_id: explicit arg > extracted from query > None
    jd_id = explicit_jd_id or extract_jd_id_from_query(query)

    # Validate jd_id exists if specified
    if jd_id:
        indexed = get_indexed_docs()
        if jd_id not in indexed["job_ids"]:
            available = ", ".join(indexed["job_ids"]) or "none"
            logger.warning("jd_id_not_found", extra={"jd_id": jd_id, "available": available})
            jd_id = None  # Fall back to un-filtered

    # Embed the query once — reused for both collection queries
    query_vec = embed_query(query)
    top_k = settings.retrieval_top_k
    lambda_mult = settings.retrieval_mmr_lambda

    resume_chunks: list[dict] = []
    jd_chunks: list[dict] = []

    gap_report = None
    ranking = None

    if intent in ("resume_lookup",):
        raw = query_resume(query_vec, top_k * 2)
        resume_chunks = mmr_rerank(raw, query_vec, lambda_mult, top_k)

    elif intent in ("jd_lookup",):
        raw = query_jobs(query_vec, top_k * 2, jd_id=jd_id)
        jd_chunks = mmr_rerank(raw, query_vec, lambda_mult, top_k)

    elif intent == "fit_score" and not jd_id:
        # Multi-JD ranking: compare resume against every indexed JD
        from app.retrieval.gap_analyser import rank_jobs, extract_manifest_from_chunks
        raw_resume = query_resume(query_vec, top_k * 2)
        resume_chunks = mmr_rerank(raw_resume, query_vec, lambda_mult, top_k)
        resume_manifest = extract_manifest_from_chunks(resume_chunks)

        indexed = get_indexed_docs()
        all_jd_data = []
        for jid in indexed["job_ids"]:
            raw_jd = query_jobs(query_vec, top_k * 2, jd_id=jid)
            jd_ch = mmr_rerank(raw_jd, query_vec, lambda_mult, top_k)
            jd_manifest = extract_manifest_from_chunks(jd_ch)
            all_jd_data.append({"jd_id": jid, "chunks": jd_ch, "manifest": jd_manifest})

        ranking = rank_jobs(resume_chunks, all_jd_data, resume_manifest)

    else:
        # gap_analysis, fit_score (specific JD), interview_prep
        from app.retrieval.gap_analyser import analyse_gap, extract_manifest_from_chunks
        raw_resume = query_resume(query_vec, top_k * 2)
        raw_jd = query_jobs(query_vec, top_k * 2, jd_id=jd_id)

        resume_chunks = mmr_rerank(raw_resume, query_vec, lambda_mult, top_k)
        jd_chunks = mmr_rerank(raw_jd, query_vec, lambda_mult, top_k)

        if intent == "gap_analysis" and resume_chunks and jd_chunks:
            resume_manifest = extract_manifest_from_chunks(resume_chunks)
            jd_manifest = extract_manifest_from_chunks(jd_chunks)
            gap_report = analyse_gap(
                resume_chunks=resume_chunks,
                jd_chunks=jd_chunks,
                jd_id=jd_id or "unknown",
                resume_manifest=resume_manifest,
                jd_manifest=jd_manifest,
            )

    logger.info(
        "retrieval_complete",
        extra={
            "intent": intent,
            "jd_id": jd_id,
            "resume_chunks": len(resume_chunks),
            "jd_chunks": len(jd_chunks),
            "gap_report": gap_report is not None,
            "ranking": ranking is not None,
        },
    )

    return {
        "intent": intent,
        "resume_chunks": resume_chunks,
        "jd_chunks": jd_chunks,
        "jd_id": jd_id,
        "gap_report": gap_report,
        "ranking": ranking,
    }
