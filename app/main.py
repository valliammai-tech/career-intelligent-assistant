"""
main.py — FastAPI application entry point.

Three route groups:
  GET  /health              — system status (Ollama, ChromaDB, indexed docs)
  POST /ingest              — upload and process a resume or JD
  POST /chat                — conversational Q&A
  GET  /docs-status         — what documents are currently indexed
  DELETE /reset             — clear all indexed data (for re-uploads)

Design decisions:
- Routes are thin — all logic lives in the service modules
- Pydantic models validate inputs at the boundary
- Errors are returned as structured JSON, never plain text
- The /health endpoint is called by Docker's HEALTHCHECK every 30s
"""
import logging
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings, Settings
from app.models import (
    IngestRequest, IngestResponse, ChatRequest, ChatResponse,
    HealthResponse, IngestedDocs, DocType, SourceChunk,
)
from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_document
from app.ingestion.embedder import embed_texts, embed_query, check_ollama_reachable
from app.ingestion.skill_extractor import extract_skills
from app.retrieval.vector_store import (
    store_resume_chunks, store_jd_chunks,
    get_store_status, get_indexed_docs,
)
from app.retrieval.retriever import retrieve_context
from app.llm.client import chat_completion
from app.llm.prompt_builder import build_messages
from app.llm.guardrails import validate_pre_call, check_grounding, add_grounding_caveat
from app.llm.memory import get_history, save_turn, clear_session
from app.observability.logger import get_logger, Timer

logger = get_logger(__name__)

# ── Upload directory ───────────────────────────────────────────────────────────
UPLOAD_DIR = Path("./data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_FILE_SIZE_MB = 10


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logger.info("app_startup", extra={"version": "1.0.0"})
    settings = get_settings()
    logger.info("config_loaded", extra={
        "llm_model": settings.ollama_llm_model,
        "embed_model": settings.ollama_embed_model,
        "chroma_dir": settings.chroma_persist_dir,
    })
    yield
    logger.info("app_shutdown")


app = FastAPI(
    title="Career Intelligence Assistant",
    description="Analyse resumes against job descriptions using local LLMs.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Streamlit frontend to call the API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://ui:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health_check():
    """
    System health check.
    Returns status of Ollama, ChromaDB, and indexed document counts.
    Used by Docker HEALTHCHECK and the Streamlit UI status bar.
    """
    ollama_ok = check_ollama_reachable()
    store = get_store_status()

    status = "ok"
    if not ollama_ok or not store["chroma_reachable"]:
        status = "degraded"

    return HealthResponse(
        status=status,
        ollama_reachable=ollama_ok,
        chroma_reachable=store["chroma_reachable"],
        resume_indexed=store["resume_indexed"],
        jobs_indexed=store["jobs_indexed"],
    )


# ── Document Status ────────────────────────────────────────────────────────────

@app.get("/docs-status", response_model=IngestedDocs, tags=["documents"])
def docs_status():
    """Return which documents are currently indexed."""
    result = get_indexed_docs()
    return IngestedDocs(**result)


@app.delete("/reset", tags=["documents"])
def reset_all():
    """
    Clear all indexed documents. Useful for starting fresh.
    WARNING: This deletes all stored vectors. Re-upload all documents after calling this.
    """
    import chromadb
    from app.config import get_settings
    from chromadb.config import Settings as ChromaSettings

    settings = get_settings()
    client = chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    for name in [settings.chroma_resume_collection, settings.chroma_jobs_collection]:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    logger.info("collections_reset")
    return {"status": "ok", "message": "All indexed documents cleared."}


# ── Ingestion ──────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse, tags=["documents"])
async def ingest_document(
    file: UploadFile = File(...),
    doc_type: str = Form(...),
    jd_id: str | None = Form(default=None),
):
    """
    Upload and ingest a document (resume or job description).

    - doc_type: "resume" or "jd"
    - jd_id: required when doc_type="jd" (e.g. "job_1", "job_2")

    Pipeline:
    1. Save uploaded file to disk
    2. Load and extract text (PyMuPDF / python-docx)
    3. Chunk using three-pass hybrid chunker
    4. Generate embeddings via Ollama nomic-embed-text
    5. Extract skills manifest via LLM
    6. Store chunks + embeddings in ChromaDB
    """
    # ── Validate inputs ──────────────────────────────────────────────────────
    try:
        doc_type_enum = DocType(doc_type)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"doc_type must be 'resume' or 'jd', got '{doc_type}'")

    if doc_type_enum == DocType.JD and not jd_id:
        raise HTTPException(status_code=422, detail="jd_id is required when doc_type is 'jd'")

    # Validate file extension
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"File type '{suffix}' not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Validate file size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum: {MAX_FILE_SIZE_MB} MB"
        )

    with Timer() as t:
        # ── Save file ────────────────────────────────────────────────────────
        safe_name = f"{doc_type}_{jd_id or 'resume'}_{file.filename}"
        file_path = UPLOAD_DIR / safe_name
        file_path.write_bytes(content)

        # ── Load ─────────────────────────────────────────────────────────────
        try:
            pages = load_document(file_path)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Failed to load document: {e}")

        if not pages:
            raise HTTPException(status_code=422, detail="Document appears to be empty or unreadable.")

        # ── Chunk ─────────────────────────────────────────────────────────────
        chunks = chunk_document(pages)
        if not chunks:
            raise HTTPException(status_code=422, detail="No content could be extracted from the document.")

        # ── Extract skills manifest (once per doc) ────────────────────────────
        full_text = "\n".join(p["text"] for p in pages)
        skills_manifest = extract_skills(full_text)

        # ── Embed ─────────────────────────────────────────────────────────────
        try:
            embeddings = embed_texts([chunk.text for chunk in chunks])
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Ollama embedding error: {e}. Is Ollama running?")

        # ── Store ─────────────────────────────────────────────────────────────
        if doc_type_enum == DocType.RESUME:
            chunks_stored = store_resume_chunks(chunks, embeddings, skills_manifest, file.filename or "resume")
        else:
            chunks_stored = store_jd_chunks(chunks, embeddings, skills_manifest, jd_id, file.filename or "jd")

    logger.info(
        "ingest_complete",
        extra={
            "doc_type": doc_type,
            "jd_id": jd_id,
            "filename": file.filename,
            "chunks": chunks_stored,
            "latency_ms": t.elapsed_ms,
        },
    )

    return IngestResponse(
        status="success",
        doc_type=doc_type_enum,
        jd_id=jd_id,
        chunks_stored=chunks_stored,
        filename=file.filename or "unknown",
        message=f"Successfully ingested {chunks_stored} chunks from '{file.filename}'.",
    )


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(request: ChatRequest):
    """
    Conversational Q&A endpoint.

    Pipeline:
    1. Classify query intent
    2. Retrieve relevant chunks (MMR from correct collection(s))
    3. Pre-call guardrail check
    4. Build composable prompt
    5. LLM completion
    6. Post-call grounding check
    7. Save turn to memory
    8. Return structured response
    """
    with Timer() as t:
        # ── Retrieve ──────────────────────────────────────────────────────────
        context = retrieve_context(query=request.query, explicit_jd_id=request.jd_id)
        intent = context["intent"]
        resume_chunks = context["resume_chunks"]
        jd_chunks = context["jd_chunks"]
        jd_id = context["jd_id"]

        # ── Pre-call guardrail ────────────────────────────────────────────────
        guard = validate_pre_call(query=request.query, intent=intent, jd_id=jd_id)
        if not guard.passed:
            return ChatResponse(
                answer=guard.reason,
                intent=intent,
                sources=[],
                session_id=request.session_id,
            )

        # ── Build prompt ──────────────────────────────────────────────────────
        history = get_history(request.session_id)
        messages = build_messages(
            query=request.query,
            intent=intent,
            resume_chunks=resume_chunks,
            jd_chunks=jd_chunks,
            jd_id=jd_id,
            conversation_history=history,
        )

        # ── LLM call ─────────────────────────────────────────────────────────
        try:
            answer, token_count = chat_completion(messages)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"LLM call failed: {e}. Is Ollama running?")

        # ── Post-call grounding check ─────────────────────────────────────────
        ground_result = check_grounding(answer, resume_chunks, jd_chunks)
        if not ground_result.passed:
            logger.warning("grounding_failed", extra={"session": request.session_id})
            answer = add_grounding_caveat(answer)

        # ── Save to memory ────────────────────────────────────────────────────
        save_turn(request.session_id, request.query, answer)

        # ── Build source citations ────────────────────────────────────────────
        sources = _build_sources(resume_chunks, jd_chunks)

    logger.info(
        "chat_complete",
        extra={
            "session_id": request.session_id,
            "intent": intent,
            "tokens": token_count,
            "latency_ms": t.elapsed_ms,
        },
    )

    return ChatResponse(
        answer=answer,
        intent=intent,
        sources=sources,
        session_id=request.session_id,
        tokens_used=token_count,
    )


def _build_sources(resume_chunks: list[dict], jd_chunks: list[dict]) -> list[SourceChunk]:
    """Convert retrieved chunks into SourceChunk objects for the response."""
    sources = []
    for chunk in resume_chunks[:3]:  # Show top 3 sources
        meta = chunk.get("metadata", {})
        sources.append(SourceChunk(
            text=chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
            doc_type=DocType.RESUME,
            section=meta.get("section", "Unknown"),
            jd_id=None,
            page=meta.get("page", 1),
        ))
    for chunk in jd_chunks[:3]:
        meta = chunk.get("metadata", {})
        sources.append(SourceChunk(
            text=chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
            doc_type=DocType.JD,
            section=meta.get("section", "Unknown"),
            jd_id=meta.get("jd_id"),
            page=meta.get("page", 1),
        ))
    return sources
