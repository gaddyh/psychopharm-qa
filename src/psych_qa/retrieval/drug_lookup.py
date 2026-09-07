"""Drug lookup — resolve drug names to canonical drugs and get claims."""

from __future__ import annotations

import logging
from typing import Any

from ..repositories import drugs as drug_repo
from ..repositories import claims as claim_repo

logger = logging.getLogger(__name__)


def lookup_drugs(drug_names: list[str]) -> dict[str, dict[str, Any]]:
    """Resolve drug names to canonical drugs and their claims.

    Returns:
        Dict mapping drug_name → {drug: {...}, claims: [...]}
    """
    results: dict[str, dict[str, Any]] = {}

    for name in drug_names:
        drug = drug_repo.find_by_name(name)
        if not drug:
            logger.warning(f"Drug not found: {name}")
            results[name] = {"drug": None, "claims": []}
            continue

        claims = claim_repo.get_claims_for_drug(drug["id"])

        # Separate by source
        nbn_claims = [c for c in claims if c["source"] == "nbn"]
        stahl_claims = [c for c in claims if c["source"] == "stahl"]

        results[name] = {
            "drug": drug,
            "nbn_claims": nbn_claims,
            "stahl_claims": stahl_claims,
            "all_claims": claims,
        }

    return results
