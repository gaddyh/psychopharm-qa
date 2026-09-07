"""Evidence builder — assembles the evidence package from all sources.

V2: uses token-budget-aware packing for Kaplan passages and removes arbitrary
truncation. Complete child chunks are sent or omitted — never cut mid-passage.
"""

from __future__ import annotations

import logging
from typing import Any

from ..domain.models import EvidenceItem, EvidencePackage, QuestionUnderstanding
from ..retrieval.chapter_search import KaplanRetriever
from ..retrieval.drug_lookup import lookup_drugs
from ..retrieval.reranker import rerank
from .token_budget import compute_kaplan_budget, pack_kaplan

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
    question_type = understanding.get("question_type")

    # 1. Deterministic drug lookup (Stahl + NbN) — filtered by question type
    drug_data = lookup_drugs(drug_names, question_type=question_type)

    # 2. Kaplan hybrid retrieval (fetch more candidates than needed for budget packing)
    retriever = KaplanRetriever(top_k=10)
    drug_ids = []
    for d in drug_data.values():
        if d.get("drug"):
            drug_ids.append(d["drug"]["id"])

    kaplan_results = retriever.retrieve(question, drug_ids=drug_ids if drug_ids else None)

    # 3. Rerank Kaplan results (no truncation — reranker sees full passages)
    kaplan_reranked = rerank(question, kaplan_results, top_k=10)

    # 4. Token-budget-aware packing: select passages that fit within budget
    # First build the deterministic evidence dict for budget computation
    deterministic_evidence: dict[str, Any] = {"drugs": {}, "conflicts_or_uncertainties": []}
    for drug_name, data in drug_data.items():
        if not data.get("drug"):
            continue
        deterministic_evidence["drugs"][drug_name] = {
            "nbn_evidence": data.get("nbn_claims", []),
            "stahl_evidence": data.get("stahl_claims", []),
        }

    # Compute Kaplan budget after accounting for deterministic evidence
    from .prompts import SYSTEM_PROMPT
    from ..llm.schemas import ANSWER_SCHEMA
    import json
    schema_json = json.dumps(ANSWER_SCHEMA)
    kaplan_budget = compute_kaplan_budget(
        system_prompt=SYSTEM_PROMPT,
        question=question,
        schema_json=schema_json,
        selected_deterministic_evidence=deterministic_evidence,
    )
    logger.info(f"Kaplan token budget: {kaplan_budget}")

    # Pack complete passages into the budget (no truncation)
    kaplan_selected, kaplan_omitted, budget_diagnostics = pack_kaplan(
        kaplan_reranked, kaplan_budget, max_results=5
    )
    logger.info(
        f"Kaplan packed: {len(kaplan_selected)} selected, {len(kaplan_omitted)} omitted "
        f"(budget {kaplan_budget}, used {budget_diagnostics['evidence_tokens_used']})"
    )

    # 5. Detect conflicts (expanded)
    conflicts = _detect_conflicts(drug_data)

    # 6. Assemble evidence package
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
        for p in kaplan_selected
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
        "kaplan_count": len(kaplan_selected),
        "kaplan_omitted_count": len(kaplan_omitted),
        "budget_diagnostics": budget_diagnostics,
    }

    return package, raw_data


def _detect_conflicts(drug_data: dict[str, dict[str, Any]]) -> list[str]:
    """Detect conflicts between NbN and Stahl claims for the same drug.

    Expanded V2: checks mechanism, dose, side effects, and indication differences.
    """
    conflicts = []

    for drug_name, data in drug_data.items():
        if not data.get("drug"):
            continue

        nbn = data.get("nbn_claims", [])
        stahl = data.get("stahl_claims", [])

        # 1. FDA approval disagreements
        stahl_fda = [c for c in stahl if c["category"] == "indication" and c.get("attributes", {}).get("fda_approved")]
        nbn_approved = [c for c in nbn if c["category"] == "indication"]

        if stahl_fda and not nbn_approved:
            conflicts.append(
                f"{drug_name}: Stahl marks indications as FDA-approved but NbN does not list FDA approval"
            )

        # 2. Mechanism disagreements
        nbn_mechanism = [c["text"].lower() for c in nbn if c["category"] in ("mode_of_action", "target", "neurotransmitter_effects")]
        stahl_mechanism = [c["text"].lower() for c in stahl if c["category"] == "mechanism"]

        # Check for presynaptic vs postsynaptic disagreements
        nbn_presynaptic = any("pre-synaptic" in m or "presynaptic" in m for m in nbn_mechanism)
        stahl_presynaptic = any("pre-synaptic" in m or "presynaptic" in m for m in stahl_mechanism)
        if nbn_presynaptic and not stahl_presynaptic:
            conflicts.append(
                f"{drug_name}: NbN describes presynaptic action but Stahl does not mention it"
            )

        # 3. Side effect disagreements
        nbn_side_effects = {c["text"].lower().strip() for c in nbn if c["category"] == "side_effects"}
        stahl_side_effects = {c["text"].lower().strip() for c in stahl if c["category"] == "side_effects"}

        # Check for serious risks mentioned in one but not the other
        serious_keywords = ["NMS", "tardive", "QTc", "prolactin", "galactorrhea"]
        for kw in serious_keywords:
            in_nbn = any(kw.lower() in s for s in nbn_side_effects)
            in_stahl = any(kw.lower() in s for s in stahl_side_effects)
            if in_nbn != in_stahl:
                conflicts.append(
                    f"{drug_name}: {kw} mentioned in {'NbN' if in_nbn else 'Stahl'} but not {'Stahl' if in_nbn else 'NbN'}"
                )

        # 4. Dose disagreements
        nbn_dose = [c["text"] for c in nbn if c["category"] == "committee_notes" and "mg" in c["text"].lower()]
        stahl_dose = [c["text"] for c in stahl if c["category"] in ("dosage_range", "dosing_instructions") and "mg" in c["text"].lower()]

        if nbn_dose and stahl_dose:
            # Check for dose range overlaps (simplified)
            import re
            nbn_doses = set()
            for d in nbn_dose:
                nbn_doses.update(re.findall(r'\d+\s*mg', d.lower()))
            stahl_doses = set()
            for d in stahl_dose:
                stahl_doses.update(re.findall(r'\d+\s*mg', d.lower()))
            if nbn_doses and stahl_doses and not nbn_doses.intersection(stahl_doses):
                conflicts.append(
                    f"{drug_name}: NbN and Stahl mention different dose values (NbN: {nbn_doses}, Stahl: {stahl_doses})"
                )

    return conflicts
