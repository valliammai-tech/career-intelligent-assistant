"""
retrieval/vector_store.py — ChromaDB dual-collection manager.

Two separate collections: 'resume' and 'jobs'.

Why separate collections?
If resume and JD chunks share one index, a query for "Python experience"
returns the highest-ranked chunks regardless of document type. A comparison
query ("How does my Python align with Job #2?") then needs post-retrieval
filtering — fragile and noisy.

With separate collections we can:
  - Run independent MMR-style retrieval per collection
  - Filter job chunks by jd_id with zero false positives
  - Scale each index independently if needed
  - Clear one collection (e.g. new resume upload) without touching the other
"""
import json
import chromadb
from chromadb import Collection
from chromadb.config import Settings as ChromaSettings
from app.config import get_settings
from app.ingestion.chunker import Chunk
from app.observability.logger import get_logger

logger = get_logger(__name__)


def _get_client() -> chromadb.ClientAPI:
    """
    Get a persistent ChromaDB client.
    PersistentClient writes to disk — data survives restarts.
    """
    settings = get_settings()
    return chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def _get_collection(name: str) -> Collection:
    """Get or create a named collection. Idempotent — safe to call on every request."""
    client = _get_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},  # Cosine similarity for text embeddings
    )


def store_resume_chunks(chunks: list[Chunk], embeddings: list[list[float]], skills_manifest: dict, filename: str) -> int:
    """
    Store resume chunks in the 'resume' collection.
    Clears any existing resume data first — we assume one resume at a time.

    Returns:
        Number of chunks stored
    """
    settings = get_settings()
    collection = _get_collection(settings.chroma_resume_collection)

    # Clear existing resume data so re-uploads don't duplicate
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        logger.info("resume_collection_cleared", extra={"deleted": len(existing["ids"])})

    _upsert_chunks(
        collection=collection,
        chunks=chunks,
        embeddings=embeddings,
        base_metadata={
            "doc_type": "resume",
            "source_file": filename,
            "skills_manifest": json.dumps(skills_manifest),
        },
    )

    logger.info("resume_stored", extra={"chunks": len(chunks), "file": filename})
    return len(chunks)


def store_jd_chunks(chunks: list[Chunk], embeddings: list[list[float]], skills_manifest: dict, jd_id: str, filename: str) -> int:
    """
    Store job description chunks in the 'jobs' collection.
    Clears existing chunks for this specific jd_id only.

    Returns:
        Number of chunks stored
    """
    settings = get_settings()
    collection = _get_collection(settings.chroma_jobs_collection)

    # Remove existing chunks for this jd_id only — other JDs are untouched
    existing = collection.get(where={"jd_id": jd_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        logger.info("jd_cleared", extra={"jd_id": jd_id, "deleted": len(existing["ids"])})

    _upsert_chunks(
        collection=collection,
        chunks=chunks,
        embeddings=embeddings,
        base_metadata={
            "doc_type": "jd",
            "jd_id": jd_id,
            "source_file": filename,
            "skills_manifest": json.dumps(skills_manifest),
        },
    )

    logger.info("jd_stored", extra={"jd_id": jd_id, "chunks": len(chunks)})
    return len(chunks)


def _upsert_chunks(
    collection: Collection,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    base_metadata: dict,
) -> None:
    """
    Insert chunks with embeddings and metadata into a collection.
    IDs are deterministic (doc_type + chunk_index) so re-ingestion is idempotent.
    """
    doc_type = base_metadata.get("doc_type", "unknown")
    jd_id = base_metadata.get("jd_id", "")

    ids = [f"{doc_type}_{jd_id}_{chunk.chunk_index}" for chunk in chunks]

    metadatas = [
        {
            **base_metadata,
            "section": chunk.section,
            "page": chunk.page,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
        }
        for chunk in chunks
    ]

    documents = [chunk.text for chunk in chunks]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


# ── Query functions ────────────────────────────────────────────────────────────

def query_resume(query_embedding: list[float], top_k: int) -> list[dict]:
    """Retrieve top_k resume chunks most similar to the query embedding."""
    settings = get_settings()
    collection = _get_collection(settings.chroma_resume_collection)
    return _query_collection(collection, query_embedding, top_k)


def query_jobs(query_embedding: list[float], top_k: int, jd_id: str | None = None) -> list[dict]:
    """
    Retrieve top_k job description chunks.
    If jd_id is specified, filters to that JD only.
    """
    settings = get_settings()
    collection = _get_collection(settings.chroma_jobs_collection)
    where = {"jd_id": jd_id} if jd_id else None
    return _query_collection(collection, query_embedding, top_k, where=where)


def _query_collection(
    collection: Collection,
    query_embedding: list[float],
    top_k: int,
    where: dict | None = None,
) -> list[dict]:
    """
    Run a similarity query against a collection.

    Returns:
        List of result dicts with keys: text, metadata, distance
    """
    count = collection.count()
    if count == 0:
        return []

    # Can't request more results than documents exist
    n_results = min(top_k, count)

    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    # Flatten the nested lists ChromaDB returns (one list per query)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(docs, metas, distances)
    ]


# ── MMR re-ranking ─────────────────────────────────────────────────────────────

def mmr_rerank(
    results: list[dict],
    query_embedding: list[float],
    lambda_mult: float,
    top_k: int,
) -> list[dict]:
    """
    Maximum Marginal Relevance re-ranking.

    Balances relevance (similarity to query) with diversity (dissimilarity
    to already-selected chunks). Prevents all results coming from one dense
    section like 'Skills' when evidence exists across multiple sections.

    lambda_mult: 0.0 = maximise diversity, 1.0 = maximise relevance
    Default 0.6 = slight preference for relevance, meaningful diversity.
    """
    if not results or top_k <= 0:
        return []

    import math

    def cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b + 1e-10)

    # ChromaDB returns distances (lower = more similar for cosine).
    # Convert to similarity scores for MMR arithmetic.
    scored = [
        {"result": r, "rel_score": 1.0 - r["distance"]}
        for r in results
    ]

    selected: list[dict] = []
    remaining = scored.copy()

    while len(selected) < top_k and remaining:
        if not selected:
            # First pick: highest relevance
            best = max(remaining, key=lambda x: x["rel_score"])
        else:
            # Subsequent picks: balance relevance vs redundancy
            selected_texts = [s["result"]["text"] for s in selected]
            best = None
            best_score = float("-inf")

            for candidate in remaining:
                rel = candidate["rel_score"]
                # Approximate redundancy as max similarity to already-selected
                # We use distance as a proxy since we don't store embeddings in results
                red = max(
                    1.0 - abs(candidate["result"]["distance"] - s["result"]["distance"])
                    for s in selected
                )
                mmr_score = lambda_mult * rel - (1 - lambda_mult) * red
                if mmr_score > best_score:
                    best_score = mmr_score
                    best = candidate

        if best is None:
            break
        selected.append(best)
        remaining.remove(best)

    return [s["result"] for s in selected]


# ── Status checks ──────────────────────────────────────────────────────────────

def get_store_status() -> dict:
    """Return counts for health check endpoint."""
    settings = get_settings()
    try:
        resume_col = _get_collection(settings.chroma_resume_collection)
        jobs_col = _get_collection(settings.chroma_jobs_collection)

        resume_count = resume_col.count()

        # Get unique jd_ids from jobs collection
        jobs_data = jobs_col.get(include=["metadatas"])
        jd_ids = list({m["jd_id"] for m in jobs_data["metadatas"] if m.get("jd_id")})

        return {
            "resume_indexed": resume_count > 0,
            "jobs_indexed": len(jd_ids),
            "chroma_reachable": True,
        }
    except Exception as e:
        logger.error("chroma_status_error", extra={"error": str(e)})
        return {"resume_indexed": False, "jobs_indexed": 0, "chroma_reachable": False}


def get_indexed_docs() -> dict:
    """Return what documents are currently indexed — for the UI status display."""
    settings = get_settings()
    try:
        resume_col = _get_collection(settings.chroma_resume_collection)
        jobs_col = _get_collection(settings.chroma_jobs_collection)

        has_resume = resume_col.count() > 0
        jobs_data = jobs_col.get(include=["metadatas"])
        job_ids = sorted(list({m["jd_id"] for m in jobs_data["metadatas"] if m.get("jd_id")}))

        return {"has_resume": has_resume, "job_ids": job_ids}
    except Exception:
        return {"has_resume": False, "job_ids": []}
