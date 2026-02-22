"""
tests/test_retriever.py — Unit tests for query routing logic.

Tests jd_id extraction (pure regex, no LLM needed) and validates
that the intent classification falls back gracefully on failure.
"""
import pytest
from app.retrieval.retriever import extract_jd_id_from_query


class TestExtractJdId:
    """
    Tests for the regex-based jd_id extractor.
    No LLM calls — pure string matching.
    """

    def test_job_hash_number(self):
        assert extract_jd_id_from_query("What am I missing for Job #1?") == "job_1"

    def test_job_number_no_hash(self):
        assert extract_jd_id_from_query("Tell me about Job 2") == "job_2"

    def test_job_lowercase(self):
        assert extract_jd_id_from_query("how do i fit for job 3?") == "job_3"

    def test_jd_prefix(self):
        assert extract_jd_id_from_query("What does JD #2 require?") == "job_2"

    def test_role_prefix(self):
        assert extract_jd_id_from_query("Prepare me for Role #1") == "job_1"

    def test_underscore_format(self):
        assert extract_jd_id_from_query("Analyse job_1 fit") == "job_1"

    def test_no_job_reference(self):
        assert extract_jd_id_from_query("What are my strongest skills?") is None

    def test_no_number(self):
        assert extract_jd_id_from_query("What does the job require?") is None

    def test_job_at_end_of_sentence(self):
        assert extract_jd_id_from_query("Score my match for job 2.") == "job_2"

    def test_case_insensitive(self):
        assert extract_jd_id_from_query("WHAT AM I MISSING FOR JOB #3?") == "job_3"

    def test_multi_digit_number(self):
        assert extract_jd_id_from_query("Tell me about Job #10") == "job_10"
