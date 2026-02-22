"""
models.py — All API request/response shapes defined as Pydantic models.

Keeping models in one file means:
- The Streamlit UI imports the same models as the API (shared contract)
- Swagger docs auto-generated from these definitions are always accurate
- Validation happens at the boundary — business logic never sees invalid data
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal
from enum import Enum


# ── Ingestion ─────────────────────────────────────────────────────────────────

class DocType(str, Enum):
    RESUME = "resume"
    JD = "jd"


class IngestRequest(BaseModel):
    doc_type: DocType
    jd_id: str | None = Field(
        default=None,
        description="Required when doc_type=jd. e.g. 'job_1', 'job_2'",
        examples=["job_1"]
    )


class IngestResponse(BaseModel):
    status: Literal["success", "error"]
    doc_type: DocType
    jd_id: str | None = None
    chunks_stored: int
    filename: str
    message: str


class IngestedDocs(BaseModel):
    """Summary of what's currently in the vector store."""
    has_resume: bool
    job_ids: list[str]


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=2000,
        description="The user's question",
        examples=["What skills am I missing for Job #1?"]
    )
    session_id: str = Field(
        description="Unique session identifier for conversation memory",
        examples=["user-abc-123"]
    )
    jd_id: str | None = Field(
        default=None,
        description="Target a specific JD. If None, the system infers from the query.",
        examples=["job_1"]
    )


class SourceChunk(BaseModel):
    """A retrieved context chunk shown to the user for transparency."""
    text: str
    doc_type: DocType
    section: str
    jd_id: str | None = None
    page: int


class ChatResponse(BaseModel):
    answer: str
    intent: str
    sources: list[SourceChunk]
    session_id: str
    tokens_used: int | None = None


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    ollama_reachable: bool
    chroma_reachable: bool
    resume_indexed: bool
    jobs_indexed: int
    version: str = "1.0.0"
