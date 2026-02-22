"""
ingestion/embedder.py — Embedding generation via Ollama.

Calls the local Ollama /api/embeddings endpoint.
Batches chunks to avoid overwhelming Ollama on large documents.

Why nomic-embed-text?
- 768-dimensional embeddings (sufficient for our use case)
- Free, runs entirely locally via Ollama
- Trained on diverse text including job postings and professional content
- Consistent quality comparable to OpenAI's ada-002

If migrating to production (OpenAI), swap the _call_ollama() function
for the OpenAI client. The rest of the pipeline is identical.
"""
import httpx
from app.config import get_settings
from app.observability.logger import get_logger

logger = get_logger(__name__)

BATCH_SIZE = 10  # Number of chunks to embed in one Ollama call


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of text strings.

    Args:
        texts: List of strings to embed

    Returns:
        List of embedding vectors (one per input text)

    Raises:
        httpx.HTTPError: if Ollama is unreachable
    """
    if not texts:
        return []

    settings = get_settings()
    all_embeddings: list[list[float]] = []

    # Process in batches — Ollama handles one text at a time,
    # so we loop but keep the batching logic here for easy swap
    # to a batch-capable provider later.
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        for text in batch:
            embedding = _embed_single(text, settings)
            all_embeddings.append(embedding)

    logger.info(
        "embeddings_generated",
        extra={"count": len(texts), "model": settings.ollama_embed_model},
    )
    return all_embeddings


def embed_query(text: str) -> list[float]:
    """Embed a single query string. Used at retrieval time."""
    settings = get_settings()
    return _embed_single(text, settings)


def _embed_single(text: str, settings) -> list[float]:
    """
    Call Ollama's embedding endpoint for a single text.

    Ollama's /api/embeddings is synchronous and handles one text per call.
    Response: {"embedding": [0.123, -0.456, ...]}
    """
    url = f"{settings.ollama_base_url}/api/embeddings"
    payload = {
        "model": settings.ollama_embed_model,
        "prompt": text,
    }
    # Generous timeout — first call loads the model into RAM (~5s on 8GB)
    with httpx.Client(timeout=60.0) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return response.json()["embedding"]


def check_ollama_reachable() -> bool:
    """Health check — verify Ollama service is up."""
    settings = get_settings()
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(settings.ollama_base_url)
            return r.status_code == 200
    except Exception:
        return False
