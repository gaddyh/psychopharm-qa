"""Citation validator — rejects nonexistent evidence IDs in LLM output.

V2: works with claims-first answer format. Every claim must have >=1
evidence_id, and every evidence_id must exist in the evidence package.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def validate_citations(
    answer: dict[str, Any],
    valid_evidence_ids: list[str],
) -> tuple[bool, list[str]]:
    """Validate that all evidence_ids in the answer's claims exist.

    Returns (is_valid, invalid_ids).
    """
    valid_set = set(valid_evidence_ids)
    invalid_ids: list[str] = []

    for claim in answer.get("claims", []):
        for eid in claim.get("evidence_ids", []):
            if eid not in valid_set:
                invalid_ids.append(eid)

    if invalid_ids:
        logger.warning(f"Invalid citation IDs: {invalid_ids}")
        return False, invalid_ids

    return True, []


def validate_every_claim_has_evidence(answer: dict[str, Any]) -> tuple[bool, list[int]]:
    """Check that every claim has at least one evidence_id.

    Returns (is_valid, claim_indices_without_evidence).
    """
    missing: list[int] = []
    for i, claim in enumerate(answer.get("claims", [])):
        eids = claim.get("evidence_ids", [])
        if not eids:
            missing.append(i)

    if missing:
        logger.warning(f"Claims without evidence_ids at indices: {missing}")
        return False, missing

    return True, []


def drop_unsupported_claims(
    answer: dict[str, Any],
    valid_evidence_ids: list[str],
) -> dict[str, Any]:
    """Drop claims where ALL evidence_ids are invalid.

    Unlike the old strip_invalid_citations, this removes the entire claim
    rather than leaving a medical claim with no support.
    """
    valid_set = set(valid_evidence_ids)
    original_claims = answer.get("claims", [])
    kept_claims: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for claim in original_claims:
        eids = claim.get("evidence_ids", [])
        valid_eids = [eid for eid in eids if eid in valid_set]
        if valid_eids:
            claim["evidence_ids"] = valid_eids
            kept_claims.append(claim)
        else:
            dropped.append(claim)

    if dropped:
        logger.info(f"Dropped {len(dropped)} claims with no valid evidence_ids")

    answer["claims"] = kept_claims
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
        "category": item.get("category", ""),
    }
    return display
