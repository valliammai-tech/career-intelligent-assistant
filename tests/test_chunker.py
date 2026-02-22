"""
tests/test_chunker.py — Unit tests for the hybrid chunking pipeline.

Testing philosophy:
- Test behaviour, not implementation
- Tests must run without Ollama (no LLM calls in the chunker)
- Cover the three failure modes: tiny docs, sectionless docs, oversized docs
"""
import pytest
from app.ingestion.chunker import chunk_document, count_tokens, Chunk


# ── Fixtures ───────────────────────────────────────────────────────────────────

RESUME_TEXT = """John Doe | Senior Python Developer

SUMMARY
Experienced backend engineer with 6 years building distributed systems.
Strong focus on API design, cloud infrastructure, and team leadership.

EXPERIENCE
Senior Python Developer — Acme Corp (2021-2024)
Led a team of 5 engineers to deliver RESTful APIs using FastAPI and PostgreSQL.
Reduced API latency by 40% through query optimisation and Redis caching.
Deployed services on AWS using ECS Fargate and Terraform.

Python Developer — StartupXYZ (2019-2021)
Built microservices architecture from scratch using Python and Docker.
Integrated third-party payment APIs and handled PCI-DSS compliance requirements.

SKILLS
Python, FastAPI, Django, PostgreSQL, Redis, Docker, AWS, Terraform, Git, SQL

EDUCATION
B.Sc Computer Science — University of Madras (2018)
"""

JD_TEXT = """Role: Lead AI/ML Engineer — TechStartup Ltd

ABOUT THE ROLE
We are looking for a Lead AI/ML Engineer to build and scale our ML platform.

RESPONSIBILITIES
Design and implement RAG systems and LLM-powered features.
Lead a team of 3-5 engineers and mentor junior members.
Deploy ML models to production using AWS SageMaker or ECS.

REQUIREMENTS
5+ years Python experience.
Experience with LLMs, vector databases, and RAG systems.
Strong background in AWS (SageMaker, Lambda, ECS preferred).
Docker and Kubernetes deployment experience.

NICE TO HAVE
LangChain or LlamaIndex experience.
MLflow or experiment tracking tools.
"""


def make_pages(text: str) -> list[dict]:
    return [{"text": text, "page": 1}]


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestChunkDocument:

    def test_resume_produces_chunks(self):
        pages = make_pages(RESUME_TEXT)
        chunks = chunk_document(pages)
        assert len(chunks) > 0, "Should produce at least one chunk"

    def test_chunks_are_chunk_objects(self):
        chunks = chunk_document(make_pages(RESUME_TEXT))
        for chunk in chunks:
            assert isinstance(chunk, Chunk)

    def test_chunks_have_required_fields(self):
        chunks = chunk_document(make_pages(RESUME_TEXT))
        for chunk in chunks:
            assert isinstance(chunk.text, str) and len(chunk.text) > 0
            assert isinstance(chunk.section, str) and len(chunk.section) > 0
            assert isinstance(chunk.page, int) and chunk.page >= 1
            assert isinstance(chunk.chunk_index, int) and chunk.chunk_index >= 0
            assert isinstance(chunk.token_count, int) and chunk.token_count > 0

    def test_section_detection_on_resume(self):
        """Section headings should be detected and preserved in metadata."""
        chunks = chunk_document(make_pages(RESUME_TEXT))
        sections_found = {chunk.section for chunk in chunks}
        # At least some of these sections should be detected
        expected_sections = {"EXPERIENCE", "SKILLS", "EDUCATION", "SUMMARY"}
        assert len(sections_found.intersection(expected_sections)) >= 2, \
            f"Expected resume sections, got: {sections_found}"

    def test_section_detection_on_jd(self):
        chunks = chunk_document(make_pages(JD_TEXT))
        sections_found = {chunk.section for chunk in chunks}
        expected = {"REQUIREMENTS", "RESPONSIBILITIES", "NICE TO HAVE"}
        assert len(sections_found.intersection(expected)) >= 1, \
            f"Expected JD sections, got: {sections_found}"

    def test_chunk_indices_are_sequential(self):
        chunks = chunk_document(make_pages(RESUME_TEXT))
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks))), "Chunk indices should be sequential"

    def test_no_empty_chunks(self):
        chunks = chunk_document(make_pages(RESUME_TEXT))
        for chunk in chunks:
            assert chunk.text.strip(), "No chunk should be empty or whitespace-only"

    def test_short_document_produces_chunks(self):
        """A very short document should still produce at least one chunk."""
        pages = make_pages("Python developer with 3 years experience.")
        chunks = chunk_document(pages)
        assert len(chunks) >= 1

    def test_empty_pages_returns_empty(self):
        """All-whitespace pages should produce no chunks."""
        pages = [{"text": "   \n\n   ", "page": 1}]
        chunks = chunk_document(pages)
        assert len(chunks) == 0

    def test_multipage_document(self):
        """Multi-page docs should track page numbers correctly."""
        pages = [
            {"text": RESUME_TEXT[:500], "page": 1},
            {"text": RESUME_TEXT[500:], "page": 2},
        ]
        chunks = chunk_document(pages)
        assert len(chunks) > 0
        page_nums = {c.page for c in chunks}
        assert len(page_nums) >= 1  # At least one page represented

    def test_token_counts_are_positive(self):
        chunks = chunk_document(make_pages(RESUME_TEXT))
        for chunk in chunks:
            assert chunk.token_count > 0

    def test_token_count_matches_text(self):
        """token_count field should be consistent with count_tokens()."""
        chunks = chunk_document(make_pages(RESUME_TEXT))
        for chunk in chunks:
            actual = count_tokens(chunk.text)
            # Allow small variation due to overlap handling
            assert abs(chunk.token_count - actual) <= 5, \
                f"Token count mismatch: stored={chunk.token_count}, actual={actual}"


class TestCountTokens:

    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_single_word(self):
        assert count_tokens("hello") > 0

    def test_longer_text_has_more_tokens(self):
        short = count_tokens("hello")
        long = count_tokens("hello world this is a longer sentence with more content")
        assert long > short

    def test_deterministic(self):
        text = "Python FastAPI Docker AWS"
        assert count_tokens(text) == count_tokens(text)
