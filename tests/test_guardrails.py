"""
tests/test_guardrails.py — Unit tests for input guardrails.

These tests mock the vector store's get_indexed_docs to avoid needing
a real ChromaDB instance. The guardrail logic is pure Python — fast and testable.
"""
import pytest
from unittest.mock import patch
from app.llm.guardrails import validate_pre_call, add_grounding_caveat, GuardResult


# ── Helpers ────────────────────────────────────────────────────────────────────

def mock_docs(has_resume: bool = True, job_ids: list | None = None):
    """Returns a patch context for get_indexed_docs."""
    return patch(
        "app.llm.guardrails.get_indexed_docs",
        return_value={"has_resume": has_resume, "job_ids": job_ids or []},
    )


# ── Pre-call validation tests ──────────────────────────────────────────────────

class TestValidatePreCall:

    def test_gap_analysis_passes_with_both_docs(self):
        with mock_docs(has_resume=True, job_ids=["job_1"]):
            result = validate_pre_call("What am I missing for job_1?", "gap_analysis", "job_1")
        assert result.passed is True

    def test_gap_analysis_fails_without_resume(self):
        with mock_docs(has_resume=False, job_ids=["job_1"]):
            result = validate_pre_call("What am I missing?", "gap_analysis", None)
        assert result.passed is False
        assert "resume" in result.reason.lower()

    def test_gap_analysis_fails_without_any_jd(self):
        with mock_docs(has_resume=True, job_ids=[]):
            result = validate_pre_call("What am I missing?", "gap_analysis", None)
        assert result.passed is False
        assert "job description" in result.reason.lower()

    def test_jd_lookup_fails_without_jds(self):
        with mock_docs(has_resume=True, job_ids=[]):
            result = validate_pre_call("What does the job require?", "jd_lookup", None)
        assert result.passed is False

    def test_resume_lookup_fails_without_resume(self):
        with mock_docs(has_resume=False, job_ids=["job_1"]):
            result = validate_pre_call("What was my last job?", "resume_lookup", None)
        assert result.passed is False
        assert "resume" in result.reason.lower()

    def test_resume_lookup_passes_with_resume(self):
        with mock_docs(has_resume=True, job_ids=[]):
            result = validate_pre_call("What was my last job?", "resume_lookup", None)
        assert result.passed is True

    def test_invalid_jd_id_fails(self):
        with mock_docs(has_resume=True, job_ids=["job_1", "job_2"]):
            result = validate_pre_call("Tell me about job_99", "jd_lookup", "job_99")
        assert result.passed is False
        assert "job_99" in result.reason

    def test_valid_jd_id_passes(self):
        with mock_docs(has_resume=True, job_ids=["job_1", "job_2"]):
            result = validate_pre_call("Tell me about job_1", "jd_lookup", "job_1")
        assert result.passed is True

    def test_empty_query_fails(self):
        with mock_docs(has_resume=True, job_ids=["job_1"]):
            result = validate_pre_call("  ", "gap_analysis", None)
        assert result.passed is False

    def test_fit_score_requires_both(self):
        with mock_docs(has_resume=True, job_ids=["job_1"]):
            result = validate_pre_call("Score my fit", "fit_score", None)
        assert result.passed is True

    def test_fit_score_fails_without_resume(self):
        with mock_docs(has_resume=False, job_ids=["job_1"]):
            result = validate_pre_call("Score my fit", "fit_score", None)
        assert result.passed is False


# ── Grounding caveat tests ─────────────────────────────────────────────────────

class TestGroundingCaveat:

    def test_caveat_is_appended(self):
        original = "You have strong Python skills."
        result = add_grounding_caveat(original)
        assert original in result
        assert "⚠️" in result
        assert len(result) > len(original)

    def test_caveat_mentions_documents(self):
        result = add_grounding_caveat("Some answer.")
        assert "document" in result.lower()

    def test_original_content_preserved(self):
        original = "Gap analysis: ✅ Python PRESENT | ❌ Kubernetes MISSING"
        result = add_grounding_caveat(original)
        assert original in result


# ── GuardResult tests ──────────────────────────────────────────────────────────

class TestGuardResult:

    def test_passed_true(self):
        g = GuardResult(passed=True, reason="OK")
        assert g.passed is True

    def test_passed_false(self):
        g = GuardResult(passed=False, reason="No resume")
        assert g.passed is False
        assert "resume" in g.reason.lower()
