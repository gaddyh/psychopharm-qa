"""Deterministic criticality assignment.

The LLM never decides whether its own claim is critical — that would be
unsafe (a model might misclassify a dosing claim as noncritical). Instead,
the application derives `critical` from the category of the cited evidence.

A claim is critical if ANY of its cited evidence is in a critical category.
"""

from __future__ import annotations

from typing import Any

# Categories where errors have direct clinical safety consequences.
CRITICAL_CATEGORIES: set[str] = {
    "dosing",
    "dosage_range",
    "dosing_instructions",
    "pregnancy",
    "breast_feeding",
    "contraindications",
    "dangerous_side_effects",
    "interaction",
    "interactions",
    "how_to_stop",
    "discontinuation",
    "overdose",
    "renal_impairment",
    "hepatic_impairment",
    "cardiac_impairment",
}


def is_critical_category(category: str | None) -> bool:
    """Check if a claim category is clinically critical."""
    if category is None:
        return False
    return category in CRITICAL_CATEGORIES


def assign_critical(
    claims: list[dict[str, Any]],
    evidence_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign the `critical` flag to each claim based on its cited evidence categories.

    Modifies claims in-place and returns them. A claim is critical if ANY
    of its cited evidence items is in a CRITICAL_CATEGORIES category.

    Args:
        claims: List of claim dicts with 'evidence_ids' field.
        evidence_lookup: Dict mapping evidence_id -> evidence item dict
                         (with 'category' field).

    Returns:
        The same claims list with 'critical' field added to each claim.
    """
    for claim in claims:
        critical = False
        for eid in claim.get("evidence_ids", []):
            evidence = evidence_lookup.get(eid)
            if evidence and is_critical_category(evidence.get("category")):
                critical = True
                break
        claim["critical"] = critical
    return claim_list_with_critical(claims)


def claim_list_with_critical(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure all claims have a 'critical' field (default False)."""
    for claim in claims:
        if "critical" not in claim:
            claim["critical"] = False
    return claims
