"""Evidence builder — assembles the evidence package from all sources."""

from __future__ import annotations

import logging
from typing import Any

from ..domain.models import EvidenceItem, EvidencePackage, QuestionUnderstanding
from ..retrieval.chapter_search import KaplanRetriever
from ..retrieval.drug_lookup import lookup_drugs
from ..retrieval.reranker import rerank

logger = logging.getLogger(__name__)


def build_evidence_package(
    question: str,
    understanding: dict[str, Any],
) -> tuple[EvidencePackage, dict[str, Any]]:
    """Build the unified evidence package from NbN, Stahl, and Kaplan.

    Returns (evidence_package, raw_lookup_data) where raw_lookup_data
    contains the drug claims and kaplan passages for trace storage.
    """
    drug_names = understanding.get("drug_names", [])

    # 1. Deterministic drug lookup (Stahl + NbN)
    drug_data = lookup_drugs(drug_names)

    # 2. Kaplan hybrid retrieval
    retriever = KaplanRetriever(top_k=10)
    drug_ids = []
    for d in drug_data.values():
        if d.get("drug"):
            drug_ids.append(d["drug"]["id"])

    kaplan_results = retriever.retrieve(question, drug_ids=drug_ids if drug_ids else None)

    # 3. Rerank Kaplan results
    kaplan_reranked = rerank(question, kaplan_results, top_k=5)

    # 4. Detect conflicts
    conflicts = _detect_conflicts(drug_data)

    # 5. Assemble evidence package
    drugs_dict: dict[str, dict[str, list[EvidenceItem]]] = {}
    for drug_name, data in drug_data.items():
        if not data.get("drug"):
            continue

        nbn_items = [
            EvidenceItem(
                evidence_id=c["evidence_id"],
                source="nbn",
                drug_id=data["drug"]["id"],
                category=c["category"],
                text=c["text"],
                attributes=c.get("attributes", {}),
                locator=c.get("locator", {}),
            )
            for c in data.get("nbn_claims", [])
        ]

        stahl_items = [
            EvidenceItem(
                evidence_id=c["evidence_id"],
                source="stahl",
                drug_id=data["drug"]["id"],
                category=c["category"],
                text=c["text"],
                attributes=c.get("attributes", {}),
                locator=c.get("locator", {}),
            )
            for c in data.get("stahl_claims", [])
        ]

        drugs_dict[drug_name] = {
            "nbn_evidence": nbn_items,
            "stahl_evidence": stahl_items,
        }

    kaplan_items = [
        EvidenceItem(
            evidence_id=p["evidence_id"],
            source="kaplan",
            text=p["text"],
            attributes={"score": p.get("fused_score", p.get("score", 0))},
            locator={"physical_pdf_page": p["physical_pdf_page"]},
        )
        for p in kaplan_reranked
    ]

    # Collect required citations (all evidence IDs)
    required = []
    for drug_name, data in drugs_dict.items():
        for e in data.get("nbn_evidence", []):
            required.append(e.evidence_id)
        for e in data.get("stahl_evidence", []):
            required.append(e.evidence_id)
    for e in kaplan_items:
        required.append(e.evidence_id)

    package = EvidencePackage(
        question=question,
        drugs=drugs_dict,
        kaplan_passages=kaplan_items,
        conflicts_or_uncertainties=conflicts,
        required_citations=required,
    )

    # Raw data for trace storage
    raw_data = {
        "drug_data": {
            name: {
                "drug": d.get("drug"),
                "nbn_count": len(d.get("nbn_claims", [])),
                "stahl_count": len(d.get("stahl_claims", [])),
            }
            for name, d in drug_data.items()
        },
        "kaplan_count": len(kaplan_reranked),
    }

    return package, raw_data


def _detect_conflicts(drug_data: dict[str, dict[str, Any]]) -> list[str]:
    """Detect conflicts between NbN and Stahl claims for the same drug."""
    conflicts = []

    for drug_name, data in drug_data.items():
        if not data.get("drug"):
            continue

        nbn = data.get("nbn_claims", [])
        stahl = data.get("stahl_claims", [])

        # Check for conflicting indications
        nbn_indications = {c["text"].lower().strip() for c in nbn if c["category"] == "indication"}
        stahl_indications = {c["text"].lower().strip() for c in stahl if c["category"] == "indication"}

        # Check for FDA approval disagreements
        stahl_fda = [c for c in stahl if c["category"] == "indication" and c.get("attributes", {}).get("fda_approved")]
        nbn_approved = [c for c in nbn if c["category"] == "indication"]

        if stahl_fda and not nbn_approved:
            conflicts.append(
                f"{drug_name}: Stahl marks indications as FDA-approved but NbN does not list FDA approval"
            )

    return conflicts
