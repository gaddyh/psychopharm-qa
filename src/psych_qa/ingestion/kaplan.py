"""Kaplan Chapter 33 PDF → hierarchical chunks → embeddings.

Parent-child chunking:
- Parent = subsection (section node in document_sections)
- Child = 400-700 model-token passages with ~100-token overlap
- Retrieve children, optionally expand to parent for context.

Entity metadata on chunks for filtering:
  {drug_ids, conditions, targets, section_path}

Tables extracted into structured chunks where possible.
Figures flagged as figure_ref (caption only, not the figure's knowledge).
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pymupdf
import tiktoken
from sqlalchemy import text as sqltext

from ..config import get_settings
from ..db.connection import get_session
from ..db.tables import (
    DocumentChunk,
    DocumentSection,
    IngestionRun,
    SourceDocument,
)

logger = logging.getLogger(__name__)

# Token counting with real tokenizer
_ENCODER = tiktoken.encoding_for_model("gpt-4o")


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _content_hash(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# Section number regex: 33.1, 33.1a, 33.13b, etc.
SECTION_RE = re.compile(r"^(33\.\d+[a-z]?)\s+(.+)")


def extract_kaplan_pages(
    pdf_path: Path, page_range: tuple[int, int]
) -> list[dict[str, Any]]:
    """Extract text from Kaplan PDF pages.

    Returns list of {page_index, text} dicts.
    """
    doc = pymupdf.open(str(pdf_path))
    pages = []
    for i in range(page_range[0], min(page_range[1], len(doc))):
        page = doc[i]
        text = page.get_text("text")
        pages.append({"page_index": i, "text": text})
        if (i - page_range[0]) % 100 == 0:
            logger.info(f"Extracted page {i}")
    doc.close()
    return pages


def detect_sections(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect section headings and build section tree.

    Uses regex for section numbers (33.1a) + font heuristics as fallback.
    """
    sections: list[dict[str, Any]] = []
    current_section_num: str | None = None

    for page in pages:
        text = page["text"]
        pdf_page = page["page_index"]

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            m = SECTION_RE.match(line)
            if m:
                section_num = m.group(1)
                title = m.group(2).strip()

                # Determine parent
                parent_id = None
                if sections:
                    # Parent is the section with the longest matching prefix
                    for s in reversed(sections):
                        if section_num.startswith(s["section_number"]):
                            parent_id = s["id"]
                            break

                section = {
                    "id": len(sections) + 1,
                    "section_number": section_num,
                    "title": title,
                    "parent_section_id": parent_id,
                    "physical_page_start": pdf_page,
                    "physical_page_end": pdf_page,
                }
                sections.append(section)
                current_section_num = section_num

            # Update end page of current section
            if sections and current_section_num:
                sections[-1]["physical_page_end"] = pdf_page

    return sections


def chunk_pages(
    pages: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    target_tokens: int = 500,
    overlap_tokens: int = 100,
    max_tokens: int = 650,
) -> list[dict[str, Any]]:
    """Chunk pages into parent (section) and child (passage) chunks.

    After initial chunking, any chunk exceeding max_tokens is split further
    by sentences until all chunks are <= max_tokens.

    Returns list of chunk dicts.
    """
    chunks: list[dict[str, Any]] = []
    chunk_idx = 0

    # Build section lookup by page range
    def find_section(pdf_page: int) -> dict | None:
        for s in sections:
            if s["physical_page_start"] <= pdf_page <= s["physical_page_end"]:
                return s
        return None

    def make_chunk(text: str, pdf_page: int, section: dict | None) -> dict[str, Any]:
        nonlocal chunk_idx
        chunk = {
            "chunk_index": chunk_idx,
            "chunk_type": "text",
            "text": text.strip(),
            "physical_pdf_page": pdf_page,
            "section_id": section["id"] if section else None,
            "section_number": section["section_number"] if section else None,
            "section_title": section["title"] if section else None,
            "input_hash": _text_hash(text),
        }
        chunk_idx += 1
        return chunk

    for page in pages:
        text = page["text"]
        pdf_page = page["page_index"]
        section = find_section(pdf_page)

        # Skip figure/table-only pages for now (flag them)
        if not text.strip():
            continue

        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            # Single-paragraph page
            paragraphs = [text.strip()]

        # Build child chunks from paragraphs
        current_text = ""
        current_tokens = 0

        for para in paragraphs:
            para_tokens = _count_tokens(para)

            if current_tokens + para_tokens > target_tokens and current_text:
                # Flush current chunk
                chunks.append(make_chunk(current_text, pdf_page, section))

                # Keep overlap from end of current text
                overlap_text = current_text[-overlap_tokens * 4 :]  # approx chars
                current_text = overlap_text + " " + para
                current_tokens = _count_tokens(current_text)
            else:
                if current_text:
                    current_text += "\n\n" + para
                else:
                    current_text = para
                current_tokens += para_tokens

        # Flush remaining
        if current_text.strip():
            chunks.append(make_chunk(current_text, pdf_page, section))

    # Post-process: split any oversized chunks by sentences
    chunks = _split_oversized_chunks(chunks, max_tokens, overlap_tokens)

    # Re-index after splitting
    for i, chunk in enumerate(chunks):
        chunk["chunk_index"] = i

    return chunks


def _split_oversized_chunks(
    chunks: list[dict[str, Any]],
    max_tokens: int,
    overlap_tokens: int,
) -> list[dict[str, Any]]:
    """Split any chunk exceeding max_tokens into smaller chunks by sentences."""
    result: list[dict[str, Any]] = []

    for chunk in chunks:
        text = chunk["text"]
        token_count = _count_tokens(text)

        if token_count <= max_tokens:
            result.append(chunk)
            continue

        # Split by sentences (keep sentence-ending punctuation)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= 1:
            # Can't split by sentences — split by tokens (hard cut)
            result.extend(_hard_split_chunk(chunk, max_tokens, overlap_tokens))
            continue

        # Greedily pack sentences into sub-chunks under max_tokens
        current_text = ""
        current_tokens = 0

        for sent in sentences:
            sent_tokens = _count_tokens(sent)
            if current_tokens + sent_tokens > max_tokens and current_text:
                # Flush sub-chunk
                sub = dict(chunk)
                sub["text"] = current_text.strip()
                sub["input_hash"] = _text_hash(current_text)
                result.append(sub)

                # Overlap: keep last few sentences
                overlap_text = current_text[-overlap_tokens * 4 :]
                current_text = overlap_text + " " + sent
                current_tokens = _count_tokens(current_text)
            else:
                if current_text:
                    current_text += " " + sent
                else:
                    current_text = sent
                current_tokens += sent_tokens

        if current_text.strip():
            sub = dict(chunk)
            sub["text"] = current_text.strip()
            sub["input_hash"] = _text_hash(current_text)
            result.append(sub)

    return result


def _hard_split_chunk(
    chunk: dict[str, Any],
    max_tokens: int,
    overlap_tokens: int,
) -> list[dict[str, Any]]:
    """Hard-split a chunk that can't be split by sentences (token-level cut)."""
    text = chunk["text"]
    # Use the encoder to split at token boundaries
    tokens = _ENCODER.encode(text)
    result: list[dict[str, Any]] = []

    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        sub_text = _ENCODER.decode(tokens[start:end])
        sub = dict(chunk)
        sub["text"] = sub_text.strip()
        sub["input_hash"] = _text_hash(sub_text)
        result.append(sub)
        if end >= len(tokens):
            break
        # Move forward with overlap
        start = end - overlap_tokens

    return result


def extract_entity_metadata(text: str, known_drugs: dict[str, int]) -> dict[str, Any]:
    """Extract drug_ids, conditions, targets from chunk text.

    Lightweight regex-based extraction for filtering, not deterministic claims.
    """
    text_lower = text.lower()
    drug_ids: list[int] = []
    conditions: list[str] = []
    targets: list[str] = []

    # Match known drug names
    for drug_name, drug_id in known_drugs.items():
        if drug_name.lower() in text_lower:
            if drug_id not in drug_ids:
                drug_ids.append(drug_id)

    # Match common conditions
    condition_patterns = [
        "schizophrenia", "depression", "bipolar", "anxiety", "psychosis",
        "mania", "dysthymia", "OCD", "PTSD", "ADHD", "insomnia",
    ]
    for cond in condition_patterns:
        if cond.lower() in text_lower:
            conditions.append(cond)

    # Match receptor targets
    target_patterns = [
        r"\bD[23]\b", r"\b5-HT\b", r"\bserotonin\b", r"\bdopamine\b",
        r"\bnorepinephrine\b", r"\bGABA\b", r"\bglutamate\b",
    ]
    for pat in target_patterns:
        if re.search(pat, text, re.IGNORECASE):
            targets.append(pat.replace(r"\b", "").replace(r"\d", ""))

    return {
        "drug_ids": drug_ids,
        "conditions": list(set(conditions)),
        "targets": list(set(targets)),
    }


def ingest_kaplan(
    pdf_path: Path | None = None,
    page_range: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Ingest Kaplan Chapter 33 into the database.

    Args:
        pdf_path: Path to Kaplan PDF. Defaults to config.
        page_range: (start, end) 0-based page indices. Defaults to a small
                    range around the antipsychotics section for POC.

    Returns:
        Summary dict with counts.
    """
    settings = get_settings()
    if pdf_path is None:
        pdf_path = settings.kaplan_pdf_path

    if not pdf_path.exists():
        raise FileNotFoundError(f"Kaplan PDF not found: {pdf_path}")

    content_hash = _content_hash(pdf_path)
    version_label = "Kaplan & Sadock's 11th Edition, Chapter 33"

    if page_range is None:
        # Full Chapter 33 range from config (PDF p.9786-11300, 0-based)
        page_range = (settings.kaplan_ch33_start_page, settings.kaplan_ch33_end_page)

    # Stage 1: Extract pages
    logger.info(f"Extracting pages {page_range[0]}-{page_range[1]} from {pdf_path.name}...")
    pages = extract_kaplan_pages(pdf_path, page_range)
    logger.info(f"Extracted {len(pages)} pages")

    # Stage 2: Detect sections
    logger.info("Detecting sections...")
    sections = detect_sections(pages)
    logger.info(f"Found {len(sections)} sections")

    # Stage 3: Chunk
    logger.info("Chunking pages...")
    chunks = chunk_pages(pages, sections)
    logger.info(f"Created {len(chunks)} chunks")

    # Get known drugs for entity metadata
    session = get_session()
    try:
        drug_rows = session.execute(
            sqltext("SELECT id, canonical_name, slug FROM drugs")
        ).fetchall()
        known_drugs = {r[1]: r[0] for r in drug_rows}  # name → id
        # Also add slug-based matching
        for r in drug_rows:
            known_drugs.setdefault(r[2], r[0])

        # Upsert source document
        doc = session.execute(
            sqltext(
                "SELECT id FROM source_documents WHERE source_type = :st AND content_hash = :ch"
            ),
            {"st": "kaplan", "ch": content_hash},
        ).fetchone()

        if doc:
            source_doc_id = doc[0]
            logger.info(f"Reusing existing source_document id={source_doc_id}")
        else:
            sd = SourceDocument(
                source_type="kaplan",
                scope_key="chapter-33",
                content_hash=content_hash,
                version_label=version_label,
                is_active=False,
            )
            session.add(sd)
            session.flush()
            source_doc_id = sd.id

        # Create ingestion run
        run = IngestionRun(
            source_document_id=source_doc_id,
            source="kaplan",
            status="running",
            started_at=datetime.utcnow(),
        )
        session.add(run)
        session.flush()
        run_id = run.id

        # Insert sections
        section_id_map: dict[int, int] = {}  # temp id → DB id
        for s in sections:
            parent_db_id = section_id_map.get(s["parent_section_id"]) if s["parent_section_id"] else None
            ds = DocumentSection(
                source_document_id=source_doc_id,
                section_number=s["section_number"],
                title=s["title"],
                parent_section_id=parent_db_id,
                physical_page_start=s["physical_page_start"],
                physical_page_end=s["physical_page_end"],
            )
            session.add(ds)
            session.flush()
            section_id_map[s["id"]] = ds.id

        # Insert chunks
        records_written = 0
        for chunk in chunks:
            # Extract entity metadata
            metadata = extract_entity_metadata(chunk["text"], known_drugs)
            if chunk.get("section_number"):
                metadata["section_path"] = [chunk["section_number"]]
                if chunk.get("section_title"):
                    metadata["section_title"] = chunk["section_title"]

            dc = DocumentChunk(
                source_document_id=source_doc_id,
                section_id=section_id_map.get(chunk["section_id"]) if chunk["section_id"] else None,
                parent_chunk_id=None,  # Set later if needed
                chunk_type=chunk["chunk_type"],
                chunk_index=chunk["chunk_index"],
                text=chunk["text"],
                table_data=None,
                entity_metadata=metadata,
                physical_pdf_page=chunk["physical_pdf_page"],
                input_hash=chunk["input_hash"],
            )
            session.add(dc)
            records_written += 1

            # Update search_vector with tsvector
            session.flush()
            session.execute(
                sqltext(
                    "UPDATE document_chunks SET search_vector = "
                    "to_tsvector('english', :text) WHERE id = :id"
                ),
                {"text": chunk["text"], "id": dc.id},
            )

        # Update run
        run.status = "completed"
        run.records_seen = len(chunks)
        run.records_written = records_written
        run.warnings = []
        run.completed_at = datetime.utcnow()

        # Atomically activate
        session.execute(
            sqltext(
                "UPDATE source_documents SET is_active = false "
                "WHERE source_type = 'kaplan' AND scope_key = 'chapter-33'"
            )
        )
        session.execute(
            sqltext("UPDATE source_documents SET is_active = true WHERE id = :id"),
            {"id": source_doc_id},
        )

        session.commit()

        logger.info(
            f"Kaplan ingestion complete: {len(sections)} sections, "
            f"{len(chunks)} chunks, {records_written} written"
        )
        return {
            "source_document_id": source_doc_id,
            "run_id": run_id,
            "sections": len(sections),
            "chunks": len(chunks),
            "records_written": records_written,
            "warnings": [],
        }

    except Exception as e:
        session.rollback()
        logger.error(f"Kaplan ingestion failed: {e}")
        raise
    finally:
        session.close()
