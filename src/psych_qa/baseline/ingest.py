"""Naive fixed-size chunking for Kaplan Chapter 33.

Token-stream chunking:
  page text → tokenize page → append token IDs to global stream
  → record page index for every token → sliding windows of 500, step 400.

A neutral separator (\\n\\n[PAGE BREAK]\\n\\n) is inserted between pages
to prevent words from merging across page boundaries. The separator
tokens are included in the stream but do not carry a page index; the
chunk's page range is derived from the real page tokens within the window.

Idempotent: re-running with the same corpus hash + pipeline version +
chunk size + overlap + embedding model either no-ops (if already ingested)
or atomically replaces the old chunks.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pymupdf
import tiktoken
from sqlalchemy import text as sqltext

from ..config import get_settings
from ..db.connection import get_session
from ..db.tables import DocumentChunk, IngestionRun, SourceDocument

logger = logging.getLogger(__name__)

_ENCODER = tiktoken.encoding_for_model("gpt-4o")
PAGE_SEPARATOR = "\n\n[PAGE BREAK]\n\n"

PIPELINE_VERSION = "baseline-v0"
SCOPE_KEY = "chapter-33-baseline"
SOURCE_TYPE = "kaplan"
DEFAULT_CHUNK_TOKENS = 500
DEFAULT_STEP_TOKENS = 400  # overlap = 100


@dataclass
class ChunkResult:
    """A single chunk produced by the tokenizer-stream chunker."""

    chunk_index: int
    text: str
    page_start: int
    page_end: int
    token_count: int
    input_hash: str


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _content_hash(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def extract_pages(pdf_path: Path, page_range: tuple[int, int] | None = None) -> list[dict[str, Any]]:
    """Extract text from all (or a range of) PDF pages.

    Returns list of {page_index, text} dicts.
    """
    doc = pymupdf.open(str(pdf_path))
    if page_range is None:
        indices = range(len(doc))
    else:
        indices = range(page_range[0], min(page_range[1], len(doc)))

    pages = []
    for i in indices:
        text = doc[i].get_text("text")
        pages.append({"page_index": i, "text": text})

    doc.close()
    return pages


def chunk_token_stream(
    pages: list[dict[str, Any]],
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    step_tokens: int = DEFAULT_STEP_TOKENS,
) -> list[ChunkResult]:
    """Chunk pages into fixed-size token windows with overlap.

    Algorithm:
      1. For each page, encode text to token IDs.
      2. Append separator tokens between pages.
      3. Record the page index for every non-separator token.
      4. Slide a window of chunk_tokens with step_tokens advance.
      5. For each window, decode tokens to text and track page range.

    Args:
        pages: List of {page_index, text} dicts.
        chunk_tokens: Window size in tokens.
        step_tokens: Step size in tokens (overlap = chunk_tokens - step_tokens).

    Returns:
        List of ChunkResult.
    """
    all_token_ids: list[int] = []
    token_pages: list[int | None] = []  # None for separator tokens

    for page in pages:
        text = page["text"]
        page_idx = page["page_index"]

        if text.strip():
            page_tokens = _ENCODER.encode(text)
            all_token_ids.extend(page_tokens)
            token_pages.extend([page_idx] * len(page_tokens))

        # Add separator between pages
        sep_tokens = _ENCODER.encode(PAGE_SEPARATOR)
        all_token_ids.extend(sep_tokens)
        token_pages.extend([None] * len(sep_tokens))

    if not all_token_ids:
        return []

    chunks: list[ChunkResult] = []
    chunk_idx = 0
    start = 0

    while start < len(all_token_ids):
        end = min(start + chunk_tokens, len(all_token_ids))
        window_tokens = all_token_ids[start:end]
        window_pages = token_pages[start:end]

        # Derive page range from real (non-None) page tokens
        real_pages = [p for p in window_pages if p is not None]
        if real_pages:
            page_start = min(real_pages)
            page_end = max(real_pages)
        else:
            # Should not happen, but guard anyway
            start += step_tokens
            continue

        text = _ENCODER.decode(window_tokens).strip()
        if text:
            chunks.append(
                ChunkResult(
                    chunk_index=chunk_idx,
                    text=text,
                    page_start=page_start,
                    page_end=page_end,
                    token_count=len(window_tokens),
                    input_hash=_text_hash(text),
                )
            )
            chunk_idx += 1

        if end >= len(all_token_ids):
            break
        start += step_tokens

    return chunks


def _find_existing_doc(session, content_hash: str) -> int | None:
    """Find an existing baseline source document by content hash."""
    row = session.execute(
        sqltext(
            "SELECT id FROM source_documents "
            "WHERE source_type = :st AND content_hash = :ch"
        ),
        {"st": SOURCE_TYPE, "ch": content_hash},
    ).fetchone()
    return row[0] if row else None


def _find_existing_chunks(session, source_doc_id: int) -> list[int]:
    """Find chunk IDs belonging to the current pipeline version."""
    rows = session.execute(
        sqltext(
            "SELECT id FROM document_chunks "
            "WHERE source_document_id = :sdid "
            "AND entity_metadata->>'pipeline' = :pipeline "
            "ORDER BY id"
        ),
        {"sdid": source_doc_id, "pipeline": PIPELINE_VERSION},
    ).fetchall()
    return [r[0] for r in rows]


def _delete_chunks_and_embeddings(session, chunk_ids: list[int]) -> None:
    """Delete chunks and their embeddings."""
    if not chunk_ids:
        return
    # Delete embeddings first
    session.execute(
        sqltext(
            "DELETE FROM document_embeddings WHERE chunk_id = ANY(:ids)"
        ),
        {"ids": chunk_ids},
    )
    session.execute(
        sqltext("DELETE FROM document_chunks WHERE id = ANY(:ids)"),
        {"ids": chunk_ids},
    )
    session.flush()


def ingest_kaplan_baseline(
    pdf_path: Path | None = None,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    step_tokens: int = DEFAULT_STEP_TOKENS,
    page_range: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Ingest Kaplan Chapter 33 with naive fixed-size chunking.

    Idempotent: if chunks with the same pipeline version already exist
    for this content hash, they are atomically replaced.

    Args:
        pdf_path: Path to the chapter-only PDF. Defaults to data/samples/.
        chunk_tokens: Window size in tokens.
        step_tokens: Step size in tokens.
        page_range: Optional (start, end) page range. Defaults to all pages.

    Returns:
        Summary dict with counts and statistics.
    """
    settings = get_settings()
    if pdf_path is None:
        pdf_path = settings.samples_dir / "kaplan_chapter33.pdf"

    if not pdf_path.exists():
        raise FileNotFoundError(f"Kaplan PDF not found: {pdf_path}")

    content_hash = _content_hash(pdf_path)
    version_label = f"Kaplan Ch.33 {PIPELINE_VERSION} naive-{chunk_tokens}"

    # Stage 1: Extract pages
    logger.info(f"Extracting pages from {pdf_path.name}...")
    pages = extract_pages(pdf_path, page_range)
    logger.info(f"Extracted {len(pages)} pages")

    # Stage 2: Chunk
    logger.info(f"Chunking with window={chunk_tokens}, step={step_tokens}...")
    chunks = chunk_token_stream(pages, chunk_tokens, step_tokens)

    # Statistics
    token_counts = [c.token_count for c in chunks]
    multi_page = sum(1 for c in chunks if c.page_start != c.page_end)
    total_tokens = sum(len(_ENCODER.encode(p["text"])) for p in pages if p["text"].strip())

    stats = {
        "pages_extracted": len(pages),
        "total_tokens": total_tokens,
        "chunks_written": len(chunks),
        "avg_chunk_tokens": sum(token_counts) / len(token_counts) if token_counts else 0,
        "min_chunk_tokens": min(token_counts) if token_counts else 0,
        "max_chunk_tokens": max(token_counts) if token_counts else 0,
        "multi_page_chunks": multi_page,
        "multi_page_pct": multi_page / len(chunks) * 100 if chunks else 0,
    }
    logger.info(
        f"Created {len(chunks)} chunks "
        f"(avg {stats['avg_chunk_tokens']:.0f} tokens, "
        f"{stats['multi_page_pct']:.1f}% multi-page)"
    )

    # Stage 3: Write to DB (idempotent)
    session = get_session()
    try:
        # Find or create source document
        source_doc_id = _find_existing_doc(session, content_hash)
        if source_doc_id:
            logger.info(f"Reusing source_document id={source_doc_id}")
        else:
            sd = SourceDocument(
                source_type=SOURCE_TYPE,
                scope_key=SCOPE_KEY,
                content_hash=content_hash,
                version_label=version_label,
                is_active=False,
            )
            session.add(sd)
            session.flush()
            source_doc_id = sd.id
            logger.info(f"Created source_document id={source_doc_id}")

        # Check for existing chunks with this pipeline version
        existing_chunk_ids = _find_existing_chunks(session, source_doc_id)
        if existing_chunk_ids:
            logger.info(f"Replacing {len(existing_chunk_ids)} existing {PIPELINE_VERSION} chunks...")
            _delete_chunks_and_embeddings(session, existing_chunk_ids)

        # Create ingestion run
        run = IngestionRun(
            source_document_id=source_doc_id,
            source=SOURCE_TYPE,
            status="running",
            started_at=datetime.utcnow(),
        )
        session.add(run)
        session.flush()
        run_id = run.id

        # Insert chunks
        for chunk in chunks:
            metadata = {
                "pipeline": PIPELINE_VERSION,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "chunk_tokens": chunk_tokens,
                "step_tokens": step_tokens,
                "token_count": chunk.token_count,
            }
            dc = DocumentChunk(
                source_document_id=source_doc_id,
                section_id=None,
                parent_chunk_id=None,
                chunk_type="text",
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                table_data=None,
                entity_metadata=metadata,
                physical_pdf_page=chunk.page_start,
                input_hash=chunk.input_hash,
            )
            session.add(dc)
            session.flush()

            # Populate search_vector
            session.execute(
                sqltext(
                    "UPDATE document_chunks SET search_vector = "
                    "to_tsvector('english', :text) WHERE id = :id"
                ),
                {"text": chunk.text, "id": dc.id},
            )

        # Update run
        run.status = "completed"
        run.records_seen = len(chunks)
        run.records_written = len(chunks)
        run.warnings = []
        run.completed_at = datetime.utcnow()

        # Atomically activate this source document
        session.execute(
            sqltext(
                "UPDATE source_documents SET is_active = false "
                "WHERE source_type = :st AND scope_key = :sk"
            ),
            {"st": SOURCE_TYPE, "sk": SCOPE_KEY},
        )
        session.execute(
            sqltext("UPDATE source_documents SET is_active = true WHERE id = :id"),
            {"id": source_doc_id},
        )

        session.commit()

        logger.info(f"Ingestion complete: {len(chunks)} chunks written")
        return {
            "source_document_id": source_doc_id,
            "run_id": run_id,
            "pipeline": PIPELINE_VERSION,
            "chunk_tokens": chunk_tokens,
            "step_tokens": step_tokens,
            **stats,
        }

    except Exception as e:
        session.rollback()
        logger.error(f"Ingestion failed: {e}")
        raise
    finally:
        session.close()
