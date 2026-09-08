"""Evidence matcher: page overlap AND anchor_text.

A retrieved chunk matches a gold evidence item if:
  1. The chunk's page range overlaps with the evidence's pdf_pages, AND
  2. At least one anchor_text phrase appears in the chunk text (after normalization).

If an anchor spans two adjacent chunks, the evidence is considered found
if either chunk alone contains it, or if two adjacent chunks together
contain it (cross-chunk anchor).

Normalization:
  - lowercase
  - collapse whitespace
  - fix hyphenated line breaks (word-\\n → word)
  - remove non-printable PDF artifacts
  - strip quote characters (straight and curly — PDF formatting, not semantic)
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from .gold_loader import EvidenceItem, GoldSet

# Quote characters to strip during normalization (PDF formatting artifacts)
_QUOTE_CHARS = set('"\'\u201c\u201d\u2018\u2019\u00ab\u00bb')


def normalize_text(text: str) -> str:
    """Normalize text for anchor matching.

    - lowercase
    - collapse whitespace
    - fix hyphenated line breaks (e.g. "seroto-\\nnin" → "serotonin")
    - remove non-printable PDF artifacts
    - strip quote characters (straight and curly)
    - normalize unicode to NFC
    """
    # Normalize unicode
    text = unicodedata.normalize("NFC", text)

    # Fix hyphenated line breaks: "word-\nword" → "wordword"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Remove non-printable characters except whitespace
    text = "".join(c for c in text if c.isprintable() or c.isspace())

    # Strip quote characters (PDF formatting, not semantic content)
    text = "".join(c for c in text if c not in _QUOTE_CHARS)

    # Lowercase
    text = text.lower()

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def pages_overlap(chunk_page_start: int, chunk_page_end: int, evidence_pages: list[int]) -> bool:
    """Check if chunk page range overlaps with evidence pages."""
    chunk_pages = set(range(chunk_page_start, chunk_page_end + 1))
    evidence_pages_set = set(evidence_pages)
    return bool(chunk_pages & evidence_pages_set)


def anchor_in_text(anchor: str, text: str) -> bool:
    """Check if a normalized anchor phrase appears in normalized text."""
    return normalize_text(anchor) in normalize_text(text)


def match_chunk_to_evidence(
    chunk: dict[str, Any],
    evidence: EvidenceItem,
) -> bool:
    """Check if a chunk matches an evidence item (page overlap AND anchor).

    Args:
        chunk: Dict with page_start, page_end, text.
        evidence: EvidenceItem with pdf_pages and anchor_text.

    Returns:
        True if page ranges overlap AND at least one anchor is found in the text.
    """
    page_start = chunk.get("page_start", chunk.get("physical_pdf_page", 0))
    page_end = chunk.get("page_end", page_start)

    if not pages_overlap(page_start, page_end, evidence.pdf_pages):
        return False

    chunk_text = chunk.get("text", "")
    for anchor in evidence.anchor_text:
        if anchor_in_text(anchor, chunk_text):
            return True

    return False


def match_chunk_to_evidence_cross(
    chunk: dict[str, Any],
    next_chunk: dict[str, Any] | None,
    evidence: EvidenceItem,
) -> bool:
    """Check if a chunk (or chunk + adjacent) matches evidence.

    First checks if the chunk alone matches. If not, checks if the
    concatenation of this chunk and the next adjacent chunk contains
    the anchor (cross-chunk anchor support).

    Args:
        chunk: Current chunk dict.
        next_chunk: The next chunk in retrieval order, or None.
        evidence: EvidenceItem.

    Returns:
        True if matched.
    """
    # Single-chunk match
    if match_chunk_to_evidence(chunk, evidence):
        return True

    # Cross-chunk: check if anchor spans this chunk and the next
    if next_chunk is None:
        return False

    page_start = chunk.get("page_start", chunk.get("physical_pdf_page", 0))
    page_end = chunk.get("page_end", page_start)
    next_page_start = next_chunk.get("page_start", next_chunk.get("physical_pdf_page", 0))

    # Only try cross-chunk if chunks are adjacent in the document
    # (not just in retrieval order — they should be from nearby pages)
    if not pages_overlap(page_start, page_end, evidence.pdf_pages):
        return False
    if not pages_overlap(
        next_page_start,
        next_chunk.get("page_end", next_page_start),
        evidence.pdf_pages,
    ):
        return False

    combined_text = chunk.get("text", "") + " " + next_chunk.get("text", "")
    for anchor in evidence.anchor_text:
        if anchor_in_text(anchor, combined_text):
            return True

    return False


def match_retrieved_to_evidence(
    retrieved_chunks: list[dict[str, Any]],
    evidence_ids: list[str],
    gold_set: GoldSet,
) -> dict[str, list[str]]:
    """Match retrieved chunks to gold evidence IDs.

    For each evidence ID, check if any retrieved chunk matches it
    (page overlap AND anchor text, with cross-chunk support).

    Args:
        retrieved_chunks: List of chunk dicts, ordered by retrieval rank.
        evidence_ids: Gold evidence IDs to match against.
        gold_set: The loaded gold set with evidence catalog.

    Returns:
        Dict mapping evidence_id → list of chunk IDs that matched it.
    """
    evidence_by_id = gold_set.evidence_by_id
    matches: dict[str, list[int]] = {}

    for eid in evidence_ids:
        if eid not in evidence_by_id:
            continue
        evidence = evidence_by_id[eid]
        matched_chunks: list[int] = []

        for i, chunk in enumerate(retrieved_chunks):
            next_chunk = retrieved_chunks[i + 1] if i + 1 < len(retrieved_chunks) else None
            if match_chunk_to_evidence_cross(chunk, next_chunk, evidence):
                matched_chunks.append(chunk["id"])

        matches[eid] = matched_chunks

    return matches


def find_matched_evidence_at_k(
    retrieved_chunks: list[dict[str, Any]],
    evidence_ids: list[str],
    gold_set: GoldSet,
    k: int,
) -> set[str]:
    """Return the set of evidence IDs matched in the top-k retrieved chunks.

    Args:
        retrieved_chunks: Full retrieved list (will be truncated to k).
        evidence_ids: Evidence IDs to check.
        gold_set: Loaded gold set.
        k: Top-k cutoff.

    Returns:
        Set of evidence IDs that were matched.
    """
    top_k = retrieved_chunks[:k]
    matches = match_retrieved_to_evidence(top_k, evidence_ids, gold_set)
    return {eid for eid, chunks in matches.items() if chunks}
