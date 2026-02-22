"""
config.py — All application settings loaded from environment variables.

Using pydantic-settings means:
- Every setting is type-validated at startup (fail fast, not at runtime)
- Defaults are documented alongside the setting
- Swapping from Ollama to OpenAI in production = change 2 env vars, zero code
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.2:3b"
    ollama_embed_model: str = "nomic-embed-text"

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_resume_collection: str = "resume"
    chroma_jobs_collection: str = "jobs"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = 5
    retrieval_mmr_lambda: float = 0.6  # 0=max diversity, 1=max relevance

    # ── LLM generation ────────────────────────────────────────────────────────
    llm_max_tokens: int = 800
    llm_temperature: float = 0.1      # low = more deterministic, better for analysis

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 400             # tokens per chunk
    chunk_overlap: int = 80           # token overlap between adjacent chunks
    chunk_min_tokens: int = 50        # drop chunks smaller than this

    # ── App ───────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    ui_port: int = 8501
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached settings instance — reads .env once at startup.
    Use get_settings() everywhere instead of instantiating Settings() directly.
    lru_cache(1) means the same object is returned on every call — no re-parsing.
    """
    return Settings()