"""Citation validator — rejects nonexistent evidence IDs in LLM output."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def validate_citations(
    answer: dict[str, Any],
    valid_evidence_ids: list[str],
) -> tuple[bool, list[str]]:
    """Validate that all evidence_ids in the answer exist in the evidence package.

    Returns (is_valid, invalid_ids).
    """
    valid_set = set(valid_evidence_ids)
    invalid_ids: list[str] = []

    for citation in answer.get("citations", []):
        for eid in citation.get("evidence_ids", []):
            if eid not in valid_set:
                invalid_ids.append(eid)

    if invalid_ids:
        logger.warning(f"Invalid citation IDs: {invalid_ids}")
        return False, invalid_ids

    return True, []


def strip_invalid_citations(answer: dict[str, Any], valid_evidence_ids: list[str]) -> dict[str, Any]:
    """Remove invalid evidence IDs from the answer (in-place fix)."""
    valid_set = set(valid_evidence_ids)
    for citation in answer.get("citations", []):
        citation["evidence_ids"] = [eid for eid in citation.get("evidence_ids", []) if eid in valid_set]
    return answer


def resolve_citation(evidence_id: str, evidence_lookup: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Resolve an evidence_id to its source info for rendering.

    Args:
        evidence_id: e.g. "stahl_claim_123"
        evidence_lookup: Dict mapping evidence_id → {source, text, locator, ...}

    Returns:
        Dict with display info (source name, page, text) or None.
    """
    item = evidence_lookup.get(evidence_id)
    if not item:
        return None

    source_names = {"nbn": "NbN", "stahl": "Stahl", "kaplan": "Kaplan Ch.33"}
    source = item.get("source", "unknown")
    locator = item.get("locator", {})

    display = {
        "source": source_names.get(source, source),
        "text": item.get("text", ""),
        "page": locator.get("printed_book_page") or locator.get("physical_pdf_page"),
    }
    return display
