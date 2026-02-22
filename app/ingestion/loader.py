"""
ingestion/loader.py — Raw document loading.

Extracts text from PDF and DOCX files, preserving page numbers.
PyMuPDF (fitz) is used for PDFs because it handles multi-column layouts
better than pypdf — important for resume parsing.

Returns a list of {text, page} dicts. Each item = one page of content.
The chunker downstream decides how to split these further.
"""
import fitz  # PyMuPDF
import docx as python_docx
from pathlib import Path
from app.observability.logger import get_logger

logger = get_logger(__name__)


def load_document(filepath: str | Path) -> list[dict]:
    """
    Load a document and return a list of page dicts.

    Returns:
        [{"text": "...", "page": 1}, {"text": "...", "page": 2}, ...]

    Raises:
        ValueError: if the file type is not supported
        FileNotFoundError: if the file does not exist
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {filepath}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages = _load_pdf(path)
    elif suffix in (".docx", ".doc"):
        pages = _load_docx(path)
    elif suffix == ".txt":
        pages = _load_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: pdf, docx, txt")

    # Filter out empty pages
    pages = [p for p in pages if p["text"].strip()]

    logger.info(
        "document_loaded",
        extra={"file": path.name, "pages": len(pages), "type": suffix},
    )
    return pages


def _load_pdf(path: Path) -> list[dict]:
    """Extract text page-by-page using PyMuPDF."""
    pages = []
    with fitz.open(str(path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            # get_text("text") preserves reading order better than default
            text = page.get_text("text")
            pages.append({"text": text, "page": page_num})
    return pages


def _load_docx(path: Path) -> list[dict]:
    """
    Extract text from DOCX.
    DOCX doesn't have real 'pages' — we treat the whole document as page 1,
    which is fine since resumes and JDs are typically 1-2 pages anyway.
    """
    doc = python_docx.Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs)
    return [{"text": text, "page": 1}]


def _load_txt(path: Path) -> list[dict]:
    """Plain text — treat as single page."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [{"text": text, "page": 1}]
