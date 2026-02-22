"""
retrieval/gap_analyser.py — Cross-collection skill gap comparison engine.

This module handles the hardest query type: comparing a resume against one
or more job descriptions at the structured skills level — not just retrieving
similar text, but reasoning about what's present, partial, or missing.

Why a dedicated module?
The retriever.py handles WHERE to fetch chunks from (routing + MMR).
The gap_analyser handles WHAT to do with those chunks once retrieved —
specifically, computing a structured skill overlap matrix that the
prompt_builder can inject as pre-computed evidence.

This separation means:
  - The LLM prompt receives structured gap data, not raw chunks alone
  - fit_score queries get a numeric overlap percentage, not a vague answer
  - Multi-JD ranking is computable without N separate LLM calls

Two analysis modes:
  1. single_jd_gap   — resume vs one specific JD (gap_analysis intent)
  2. multi_jd_rank   — resume vs all indexed JDs (fit_score intent)
"""
import json
import math
from dataclasses import dataclass
from app.observability.logger import get_logger

logger = get_logger(__name__)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class SkillGap:
    """Result of comparing one skill from the JD against the resume."""
    skill: str
    status: str          # "present" | "partial" | "missing"
    evidence: str        # Quote or summary from the resume supporting the status
    confidence: float    # 0.0–1.0 based on embedding similarity


@dataclass
class GapReport:
    """Full gap analysis result for one resume vs one JD."""
    jd_id: str
    present: list[SkillGap]
    partial: list[SkillGap]
    missing: list[SkillGap]
    fit_score: float        # 0–100 numeric score
    score_breakdown: dict   # Detailed score components


@dataclass
class RankEntry:
    """One entry in a multi-JD ranking."""
    jd_id: str
    fit_score: float
    top_strengths: list[str]
    top_gaps: list[str]


# ── Core gap analysis ──────────────────────────────────────────────────────────

def analyse_gap(
    resume_chunks: list[dict],
    jd_chunks: list[dict],
    jd_id: str,
    resume_manifest: dict,
    jd_manifest: dict,
) -> GapReport:
    """
    Compute a structured gap analysis between a resume and one JD.

    Strategy:
    1. Use the skills manifests (pre-extracted at ingest) as the structured
       skill lists — these are more reliable than re-extracting from chunks
    2. For each JD required skill, compute cosine similarity against each
       resume skill using a simple token-overlap heuristic (no extra embedding
       call needed — we already have the manifests)
    3. Classify each skill as present / partial / missing based on similarity
    4. Compute a weighted fit score

    Args:
        resume_chunks: Retrieved resume chunks from ChromaDB
        jd_chunks:     Retrieved JD chunks from ChromaDB
        jd_id:         The JD identifier string
        resume_manifest: Extracted skills dict from the resume
        jd_manifest:     Extracted skills dict from the JD

    Returns:
        GapReport with structured skill gaps and fit score
    """
    # Flatten all JD required skills into one list for comparison
    jd_skills = (
        jd_manifest.get("hard_skills", []) +
        jd_manifest.get("tools_and_technologies", [])
    )
    jd_soft = jd_manifest.get("soft_skills", [])

    # Flatten all resume skills
    resume_skills = (
        resume_manifest.get("hard_skills", []) +
        resume_manifest.get("tools_and_technologies", [])
    )
    resume_soft = resume_manifest.get("soft_skills", [])
    resume_text = _chunks_to_text(resume_chunks)

    # ── Technical skill matching ───────────────────────────────────────────────
    present: list[SkillGap] = []
    partial: list[SkillGap] = []
    missing: list[SkillGap] = []

    for jd_skill in jd_skills:
        gap = _classify_skill(jd_skill, resume_skills, resume_text)
        if gap.status == "present":
            present.append(gap)
        elif gap.status == "partial":
            partial.append(gap)
        else:
            missing.append(gap)

    # ── Soft skill matching ────────────────────────────────────────────────────
    for soft_skill in jd_soft:
        gap = _classify_skill(soft_skill, resume_soft, resume_text)
        if gap.status == "present":
            present.append(gap)
        elif gap.status == "partial":
            partial.append(gap)
        else:
            missing.append(gap)

    # ── Fit score calculation ─────────────────────────────────────────────────
    score, breakdown = _compute_fit_score(
        present=present,
        partial=partial,
        missing=missing,
        resume_manifest=resume_manifest,
        jd_manifest=jd_manifest,
    )

    report = GapReport(
        jd_id=jd_id,
        present=present,
        partial=partial,
        missing=missing,
        fit_score=score,
        score_breakdown=breakdown,
    )

    logger.info(
        "gap_analysis_complete",
        extra={
            "jd_id": jd_id,
            "present": len(present),
            "partial": len(partial),
            "missing": len(missing),
            "fit_score": score,
        },
    )
    return report


def rank_jobs(
    resume_chunks: list[dict],
    all_jd_results: list[dict],     # [{jd_id, chunks, manifest}, ...]
    resume_manifest: dict,
) -> list[RankEntry]:
    """
    Rank all indexed JDs by fit score against the resume.

    Called for fit_score intent when no specific jd_id is given.
    Returns entries sorted by fit_score descending.

    Args:
        resume_chunks:   Retrieved resume chunks
        all_jd_results:  List of {jd_id, chunks, manifest} dicts for each JD
        resume_manifest: Resume skills manifest

    Returns:
        List of RankEntry sorted best-fit first
    """
    rankings: list[RankEntry] = []

    for jd_data in all_jd_results:
        jd_id = jd_data["jd_id"]
        jd_manifest = jd_data.get("manifest", {})
        jd_chunks = jd_data.get("chunks", [])

        report = analyse_gap(
            resume_chunks=resume_chunks,
            jd_chunks=jd_chunks,
            jd_id=jd_id,
            resume_manifest=resume_manifest,
            jd_manifest=jd_manifest,
        )

        top_strengths = [g.skill for g in report.present[:3]]
        top_gaps = [g.skill for g in report.missing[:3]]

        rankings.append(RankEntry(
            jd_id=jd_id,
            fit_score=report.fit_score,
            top_strengths=top_strengths,
            top_gaps=top_gaps,
        ))

    rankings.sort(key=lambda r: r.fit_score, reverse=True)
    return rankings


def format_gap_report_for_prompt(report: GapReport) -> str:
    """
    Format a GapReport as a compact string for injection into the LLM prompt.

    This pre-computed summary lets the LLM focus on narrative and nuance
    rather than re-deriving what's present/missing from raw chunks.
    It is injected into the CONTEXT BLOCK of the system prompt.
    """
    lines = [
        f"--- PRE-COMPUTED GAP ANALYSIS: {report.jd_id} ---",
        f"Overall Fit Score: {report.fit_score:.0f}/100",
        "",
    ]

    if report.present:
        lines.append("✅ PRESENT SKILLS:")
        for gap in report.present[:8]:
            lines.append(f"  • {gap.skill} — {gap.evidence[:80]}")

    if report.partial:
        lines.append("")
        lines.append("⚠️ PARTIAL MATCH:")
        for gap in report.partial[:5]:
            lines.append(f"  • {gap.skill} — {gap.evidence[:80]}")

    if report.missing:
        lines.append("")
        lines.append("❌ MISSING / NOT FOUND:")
        for gap in report.missing[:8]:
            lines.append(f"  • {gap.skill}")

    lines.append("")
    lines.append(f"Score breakdown: {json.dumps(report.score_breakdown)}")
    lines.append("--- END GAP ANALYSIS ---")

    return "\n".join(lines)


def format_ranking_for_prompt(rankings: list[RankEntry]) -> str:
    """Format multi-JD ranking for prompt injection."""
    lines = ["--- PRE-COMPUTED JOB RANKING ---"]
    for i, entry in enumerate(rankings, start=1):
        lines.append(f"{i}. {entry.jd_id}: {entry.fit_score:.0f}/100")
        if entry.top_strengths:
            lines.append(f"   Strengths: {', '.join(entry.top_strengths)}")
        if entry.top_gaps:
            lines.append(f"   Gaps: {', '.join(entry.top_gaps)}")
    lines.append("--- END RANKING ---")
    return "\n".join(lines)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _classify_skill(
    jd_skill: str,
    resume_skills: list[str],
    resume_full_text: str,
) -> SkillGap:
    """
    Classify a JD skill as present / partial / missing against the resume.

    Uses a three-tier matching strategy:
    1. Exact match (case-insensitive) against resume skills list → present
    2. Token overlap similarity ≥ 0.5 → partial
    3. Keyword found in full resume text → partial (contextual mention)
    4. No match → missing

    This avoids an extra embedding call per skill while still catching
    partial matches like "AWS Lambda" matching "AWS" in the resume.
    """
    jd_skill_lower = jd_skill.lower().strip()
    resume_skills_lower = [s.lower().strip() for s in resume_skills]

    # Tier 1: Exact match
    if jd_skill_lower in resume_skills_lower:
        return SkillGap(
            skill=jd_skill,
            status="present",
            evidence=f"Listed directly in resume skills: '{jd_skill}'",
            confidence=1.0,
        )

    # Tier 2: Token overlap similarity (handles "AWS Lambda" vs "AWS")
    best_sim, best_match = _best_token_overlap(jd_skill_lower, resume_skills_lower)
    if best_sim >= 0.6:
        return SkillGap(
            skill=jd_skill,
            status="present",
            evidence=f"Matched to '{resume_skills[resume_skills_lower.index(best_match)]}' (similarity: {best_sim:.0%})",
            confidence=best_sim,
        )
    if best_sim >= 0.3:
        return SkillGap(
            skill=jd_skill,
            status="partial",
            evidence=f"Partial match to '{resume_skills[resume_skills_lower.index(best_match)]}' in resume skills",
            confidence=best_sim,
        )

    # Tier 3: Keyword mention in full resume text (contextual evidence)
    if _keyword_in_text(jd_skill_lower, resume_full_text.lower()):
        return SkillGap(
            skill=jd_skill,
            status="partial",
            evidence=f"'{jd_skill}' mentioned in resume context but not listed as a primary skill",
            confidence=0.4,
        )

    # Tier 4: Not found
    return SkillGap(
        skill=jd_skill,
        status="missing",
        evidence="Not found in resume skills or experience sections",
        confidence=0.0,
    )


def _best_token_overlap(target: str, candidates: list[str]) -> tuple[float, str]:
    """
    Find the candidate string with the highest token-overlap similarity to target.

    Token overlap = |intersection(target_tokens, candidate_tokens)| /
                    |union(target_tokens, candidate_tokens)|  (Jaccard similarity)
    """
    if not candidates:
        return 0.0, ""

    target_tokens = set(target.split())
    best_sim = 0.0
    best_match = ""

    for candidate in candidates:
        cand_tokens = set(candidate.split())
        if not target_tokens and not cand_tokens:
            continue
        intersection = len(target_tokens & cand_tokens)
        union = len(target_tokens | cand_tokens)
        sim = intersection / union if union > 0 else 0.0
        if sim > best_sim:
            best_sim = sim
            best_match = candidate

    return best_sim, best_match


def _keyword_in_text(keyword: str, text: str) -> bool:
    """Check if any token of the keyword phrase appears as a word in text."""
    # Check the full phrase first
    if keyword in text:
        return True
    # Check if the primary keyword token (longest word) appears
    tokens = [t for t in keyword.split() if len(t) > 3]  # skip short words
    return any(token in text for token in tokens)


def _compute_fit_score(
    present: list[SkillGap],
    partial: list[SkillGap],
    missing: list[SkillGap],
    resume_manifest: dict,
    jd_manifest: dict,
) -> tuple[float, dict]:
    """
    Compute a 0–100 fit score from gap analysis results.

    Scoring weights:
    - Technical skill coverage: 60 points
    - Soft skill coverage:       20 points
    - Experience years match:    20 points

    Within technical/soft:
    - present skill  = full weight
    - partial skill  = half weight
    - missing skill  = zero weight
    """
    # Separate technical and soft skills
    all_jd_hard = set(
        jd_manifest.get("hard_skills", []) +
        jd_manifest.get("tools_and_technologies", [])
    )
    all_jd_soft = set(jd_manifest.get("soft_skills", []))

    total_hard = len(all_jd_hard)
    total_soft = len(all_jd_soft)

    # Count present/partial for hard skills
    hard_present = sum(1 for g in present if g.skill in all_jd_hard)
    hard_partial = sum(1 for g in partial if g.skill in all_jd_hard)

    # Count present/partial for soft skills
    soft_present = sum(1 for g in present if g.skill in all_jd_soft)
    soft_partial = sum(1 for g in partial if g.skill in all_jd_soft)

    # Technical score (60 points)
    if total_hard > 0:
        tech_raw = (hard_present + 0.5 * hard_partial) / total_hard
        tech_score = tech_raw * 60
    else:
        tech_score = 30.0  # No hard skills listed = neutral

    # Soft skill score (20 points)
    if total_soft > 0:
        soft_raw = (soft_present + 0.5 * soft_partial) / total_soft
        soft_score = soft_raw * 20
    else:
        soft_score = 10.0  # No soft skills listed = neutral

    # Experience years score (20 points)
    resume_exp = resume_manifest.get("experience_years") or 0
    jd_exp_required = jd_manifest.get("experience_years") or 0

    if jd_exp_required > 0:
        exp_ratio = min(resume_exp / jd_exp_required, 1.2)  # cap at 120%
        exp_score = min(exp_ratio * 20, 20)
    else:
        exp_score = 10.0  # No experience requirement = neutral

    total = round(tech_score + soft_score + exp_score, 1)
    total = max(0.0, min(100.0, total))  # Clamp to [0, 100]

    breakdown = {
        "technical_skills": round(tech_score, 1),
        "soft_skills": round(soft_score, 1),
        "experience_years": round(exp_score, 1),
        "total": total,
    }

    return total, breakdown


def _chunks_to_text(chunks: list[dict]) -> str:
    """Flatten chunk list to a single text string for keyword searching."""
    return " ".join(c.get("text", "") for c in chunks)


# ── Manifest extraction helper ─────────────────────────────────────────────────

def extract_manifest_from_chunks(chunks: list[dict]) -> dict:
    """
    Extract the skills manifest from the first chunk's metadata.

    The manifest was stored at ingest time by skill_extractor.py.
    If missing (e.g. old data), returns an empty manifest.
    """
    from app.ingestion.skill_extractor import EMPTY_MANIFEST
    if not chunks:
        return EMPTY_MANIFEST.copy()

    manifest_raw = chunks[0].get("metadata", {}).get("skills_manifest", "{}")
    try:
        return json.loads(manifest_raw)
    except Exception:
        return EMPTY_MANIFEST.copy()
