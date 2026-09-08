"""Evidence matchers — resolve stable AcceptableEvidence to actual EvidenceItems.

Matches by (source, drug_slug, category, claim_hash) for Stahl/NbN claims
and by locator for Kaplan passages. Falls back to text_contains matching.

This is how Recall@K is computed without brittle database IDs.
"""

from __future__ import annotations

from typing import Any

from .schemas import AcceptableEvidence, RequiredAnswerPoint


def _normalize_source(source: str) -> str:
    """Normalize source name for comparison."""
    return source.lower().strip()


def match_evidence_item(
    acceptable: AcceptableEvidence,
    evidence_item: dict[str, Any],
    drug_slug_lookup: dict[int, str] | None = None,
) -> bool:
    """Check if an evidence item matches an acceptable evidence spec.

    Args:
        acceptable: The stable evidence spec from a gold case.
        evidence_item: An evidence item from an evidence package (dict with
                       evidence_id, source, category, text, attributes, locator).
        drug_slug_lookup: Optional map of drug_id -> slug for resolving evidence items.

    Returns:
        True if the evidence item matches the acceptable spec.
    """
    # Source must match
    if _normalize_source(evidence_item.get("source", "")) != _normalize_source(acceptable.source):
        return False

    # Category match (for stahl/nbn)
    if acceptable.category is not None:
        if evidence_item.get("category") != acceptable.category:
            return False

    # claim_hash match (most stable)
    if acceptable.claim_hash is not None:
        item_attrs = evidence_item.get("attributes", {})
        item_hash = item_attrs.get("claim_hash")
        if item_hash is not None and item_hash == acceptable.claim_hash:
            return True
        # Also check locator for claim_hash
        item_locator = evidence_item.get("locator", {})
        if item_locator.get("claim_hash") == acceptable.claim_hash:
            return True

    # drug_slug match
    if acceptable.drug_slug is not None and drug_slug_lookup:
        item_drug_id = evidence_item.get("drug_id")
        if item_drug_id is not None:
            item_slug = drug_slug_lookup.get(item_drug_id)
            if item_slug != acceptable.drug_slug:
                return False

    # locator match (for kaplan)
    if acceptable.locator is not None:
        item_locator = evidence_item.get("locator", {})
        for key, val in acceptable.locator.items():
            if item_locator.get(key) != val:
                return False
        return True

    # text_contains fallback
    if acceptable.text_contains is not None:
        item_text = evidence_item.get("text", "")
        if acceptable.text_contains.lower() in item_text.lower():
            return True

    # If we only matched source + category, that's a weak match
    if acceptable.claim_hash is None and acceptable.locator is None and acceptable.text_contains is None:
        # Source + category only match (weak but valid for some specs)
        return True

    return False


def resolve_acceptable_evidence(
    acceptable: AcceptableEvidence,
    evidence_package: dict[str, Any],
    drug_slug_lookup: dict[int, str] | None = None,
) -> list[str]:
    """Find all evidence IDs in the package that match an acceptable spec.

    Returns:
        List of matching evidence_id strings.
    """
    matches: list[str] = []

    # Check drug-based evidence (stahl/nbn)
    for drug_name, drug_data in evidence_package.get("drugs", {}).items():
        for source_key in ("nbn_evidence", "stahl_evidence"):
            for item in drug_data.get(source_key, []):
                if match_evidence_item(acceptable, item, drug_slug_lookup):
                    matches.append(item["evidence_id"])

    # Check Kaplan passages
    for item in evidence_package.get("kaplan_passages", []):
        if match_evidence_item(acceptable, item, drug_slug_lookup):
            matches.append(item["evidence_id"])

    return matches


def compute_recall_at_k(
    required_points: list[RequiredAnswerPoint],
    evidence_package: dict[str, Any],
    drug_slug_lookup: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Compute Recall@K for required answer points.

    For each required answer point, check if at least one of its acceptable
    evidence items was found in the evidence package.

    Returns:
        Dict with:
        - recall: fraction of points with at least one acceptable evidence found
        - points_covered: count of points with evidence
        - points_total: total required points
        - per_point: list of {point_text, found, matched_evidence_ids}
    """
    per_point: list[dict[str, Any]] = []
    points_covered = 0

    for point in required_points:
        matched_ids: list[str] = []
        for acceptable in point.acceptable_evidence:
            ids = resolve_acceptable_evidence(acceptable, evidence_package, drug_slug_lookup)
            matched_ids.extend(ids)

        found = len(set(matched_ids)) > 0
        if found:
            points_covered += 1

        per_point.append({
            "point_text": point.text,
            "found": found,
            "matched_evidence_ids": list(set(matched_ids)),
        })

    total = len(required_points)
    recall = points_covered / total if total > 0 else 1.0

    return {
        "recall": recall,
        "points_covered": points_covered,
        "points_total": total,
        "per_point": per_point,
    }


def build_drug_slug_lookup() -> dict[int, str]:
    """Build a drug_id -> slug lookup from the database."""
    from sqlalchemy import text
    from ..db.connection import get_engine

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT id, slug FROM drugs")).fetchall()
    return {r[0]: r[1] for r in rows}
