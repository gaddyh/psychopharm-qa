"""Sufficiency checker — determines if evidence is sufficient to answer."""

from __future__ import annotations

from typing import Any


def check_sufficiency(evidence_package: dict[str, Any]) -> tuple[bool, str | None]:
    """Check if the evidence package contains enough evidence to answer.

    Returns (is_sufficient, reason_if_not).
    """
    total_claims = 0
    total_passages = 0

    for drug_name, drug_data in evidence_package.get("drugs", {}).items():
        nbn = drug_data.get("nbn_evidence", [])
        stahl = drug_data.get("stahl_evidence", [])
        total_claims += len(nbn) + len(stahl)

    total_passages = len(evidence_package.get("kaplan_passages", []))

    # Need at least some deterministic evidence
    if total_claims == 0 and total_passages == 0:
        return False, "No evidence found for the requested drug(s)."

    # If no drug was found at all
    if not evidence_package.get("drugs"):
        return False, "No recognized drugs in the question."

    return True, None
