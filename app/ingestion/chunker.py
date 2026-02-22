"""
ingestion/chunker.py — Three-pass hybrid chunking strategy.

Why not a simple fixed-size splitter?
1. Resume sections have wildly different semantic density.
   Treating "Senior Python Developer 2021-2024, Led team of 12..."
   and "Skills: Python, FastAPI, Docker" as equivalent is wrong.
2. JD structure (responsibilities vs requirements vs nice-to-haves) maps
   directly to query intent. Losing that structure loses retrieval precision.
3. A sentence split from its context ("Managed a team" vs "Managed a team
   of 12 engineers") loses meaning for gap-analysis queries.

Three passes:
  Pass 1 — Detect section boundaries (regex on headings)
  Pass 2 — Split within sections using recursive character splitting
  Pass 3 — Enforce min size, merge tiny adjacent fragments
"""
import re
import tiktoken
from dataclasses import dataclass
from app.config import get_settings
from app.observability.logger import get_logger

logger = get_logger(__name__)

# Section heading patterns that work for both resumes and job descriptions
SECTION_PATTERNS = re.compile(
    r"^(EXPERIENCE|WORK EXPERIENCE|EMPLOYMENT|EDUCATION|SKILLS|TECHNICAL SKILLS|"
    r"SUMMARY|OBJECTIVE|PROFILE|CERTIFICATIONS|PROJECTS|RESPONSIBILITIES|"
    r"REQUIREMENTS|QUALIFICATIONS|NICE TO HAVE|PREFERRED|BENEFITS|"
    r"ABOUT THE ROLE|ABOUT US|WHO YOU ARE|WHAT YOU.LL DO|WHAT WE.RE LOOKING FOR)",
    re.IGNORECASE | re.MULTILINE,
)

# Token encoder — we use cl100k_base (same as GPT-4) as a consistent approximation.
# nomic-embed-text uses a different tokeniser but the token counts are close enough
# for chunking purposes. The important thing is consistency.
_ENCODER = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    text: str
    section: str        # which resume/JD section this came from
    page: int           # source page number
    chunk_index: int    # position within the document
    token_count: int    # approximate token count


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def chunk_document(pages: list[dict]) -> list[Chunk]:
    """
    Main entry point. Takes output from loader.py and returns Chunk objects.

    Args:
        pages: [{"text": "...", "page": 1}, ...]

    Returns:
        List of Chunk objects ready for embedding and storage
    """
    settings = get_settings()

    # Combine all pages into one text block with page markers
    # We track page boundaries so metadata stays accurate
    combined = "\n\n".join(
        f"[PAGE {p['page']}]\n{p['text']}" for p in pages
    )

    # Pass 1: split into sections
    sections = _detect_sections(combined)

    # Pass 2: split each section into sized chunks
    raw_chunks: list[tuple[str, str, int]] = []  # (text, section, page)
    for section_name, section_text, page_num in sections:
        splits = _split_section(section_text, settings.chunk_size, settings.chunk_overlap)
        for split in splits:
            raw_chunks.append((split, section_name, page_num))

    # Pass 3: enforce minimum size, drop noise fragments
    final_chunks = _enforce_min_size(raw_chunks, settings.chunk_min_tokens)

    chunks = [
        Chunk(
            text=text,
            section=section,
            page=page,
            chunk_index=i,
            token_count=count_tokens(text),
        )
        for i, (text, section, page) in enumerate(final_chunks)
    ]

    logger.info(
        "chunking_complete",
        extra={"raw_chunks": len(raw_chunks), "final_chunks": len(chunks)},
    )
    return chunks


def _detect_sections(text: str) -> list[tuple[str, str, int]]:
    """
    Pass 1: Split document into named sections.

    Returns:
        [(section_name, section_text, page_number), ...]
    """
    # Find all section heading positions
    matches = list(SECTION_PATTERNS.finditer(text))

    if not matches:
        # No recognisable sections — treat entire document as one section
        page = _extract_page_num(text[:50])
        return [("GENERAL", text, page)]

    sections = []
    for i, match in enumerate(matches):
        section_name = match.group().strip().upper()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        page = _extract_page_num(section_text[:50])
        sections.append((section_name, section_text, page))

    # Also capture any text before the first detected section
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.insert(0, ("HEADER", preamble, 1))

    return sections


def _split_section(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Pass 2: Recursively split a section into token-bounded chunks.

    Tries splitting on paragraph breaks first, then sentences, then words.
    This preserves semantic coherence better than hard character cuts.
    """
    tokens = count_tokens(text)
    if tokens <= chunk_size:
        return [text]

    # Try splitting on double newlines (paragraph breaks) first
    splits = _recursive_split(
        text,
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=chunk_size,
        overlap=overlap,
    )
    return splits


def _recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """
    Split text using the first separator that creates chunks within chunk_size.
    Falls back to next separator if chunks are still too large.
    """
    if not separators:
        # Hard cut by characters as last resort
        return _hard_cut(text, chunk_size, overlap)

    sep = separators[0]
    parts = text.split(sep)

    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = (current + sep + part).strip() if current else part.strip()
        if count_tokens(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # Part itself might be too large — recurse with next separator
            if count_tokens(part) > chunk_size:
                sub_chunks = _recursive_split(part, separators[1:], chunk_size, overlap)
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = part.strip()

    if current:
        chunks.append(current)

    # Add overlap: prepend the tail of the previous chunk to the next
    if overlap > 0 and len(chunks) > 1:
        chunks = _add_overlap(chunks, overlap)

    return chunks


def _add_overlap(chunks: list[str], overlap_tokens: int) -> list[str]:
    """Prepend the tail of each chunk to the next chunk for context continuity."""
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tokens = _ENCODER.encode(chunks[i - 1])
        overlap_text = _ENCODER.decode(prev_tokens[-overlap_tokens:])
        result.append((overlap_text + " " + chunks[i]).strip())
    return result


def _hard_cut(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Last-resort character cut when all separator strategies fail."""
    tokens = _ENCODER.encode(text)
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(tokens), step):
        chunk_tokens = tokens[start : start + chunk_size]
        chunks.append(_ENCODER.decode(chunk_tokens))
    return chunks


def _enforce_min_size(
    raw_chunks: list[tuple[str, str, int]],
    min_tokens: int,
) -> list[tuple[str, str, int]]:
    """
    Pass 3: Drop chunks below min_tokens. Merge tiny fragments into
    the previous chunk if they're from the same section.
    """
    result: list[tuple[str, str, int]] = []
    for text, section, page in raw_chunks:
        tokens = count_tokens(text)
        if tokens < min_tokens:
            # Merge into previous chunk if same section, otherwise drop
            if result and result[-1][1] == section:
                prev_text, prev_section, prev_page = result[-1]
                result[-1] = (prev_text + " " + text, prev_section, prev_page)
            # else: silently drop the tiny fragment
        else:
            result.append((text, section, page))
    return result


def _extract_page_num(text_prefix: str) -> int:
    """Extract [PAGE N] marker if present, default to 1."""
    match = re.search(r"\[PAGE (\d+)\]", text_prefix)
    return int(match.group(1)) if match else 1
