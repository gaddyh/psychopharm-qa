"""Sufficiency checker — question-specific rules for abstention.

V2: implements per-question-type sufficiency rules. The system abstains
when the required evidence for a question type is not available.

Rules:
- dosing: requires dosing evidence (dosage_range, dosing_instructions, or dosing)
- interactions: requires an interaction claim involving the drug(s)
- indication/FDA: requires explicit indication or regulatory-status evidence
- mechanism: requires target/action evidence (mechanism, mode_of_action, target)
- special_populations: requires pregnancy/renal/hepatic/elderly/pediatric evidence
- patient_specific: requires clarification (always needs_clarification)
- false_premise: abstain with correction (detected via premises check)
- comparison: requires evidence for all compared drugs
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Categories required by each question type
REQUIRED_CATEGORIES: dict[str, list[set[str]]] = {
    # Each entry is a list of alternative category sets — at least one set must be fully present
    "dosing": [
        {"dosage_range", "dosing_instructions"},
        {"dosing"},
    ],
    "interactions": [
        {"interactions"},
    ],
    "indication": [
        {"indication"},
        {"efficacy"},
    ],
    "mechanism": [
        {"mechanism", "target"},
        {"mode_of_action", "target"},
        {"mechanism"},
        {"mode_of_action"},
    ],
    "special_populations": [
        {"pregnancy"},
        {"renal_impairment"},
        {"hepatic_impairment"},
        {"elderly"},
        {"pediatric"},
    ],
    "pharmacokinetics": [
        {"pharmacokinetics"},
    ],
    "side_effects": [
        {"side_effects"},
    ],
}


def check_sufficiency(
    evidence_package: dict[str, Any],
    understanding: dict[str, Any] | None = None,
) -> tuple[bool, str | None]:
    """Check if the evidence package contains enough evidence to answer.

    Returns (is_sufficient, reason_if_not).
    """
    # Basic check: no drugs found
    if not evidence_package.get("drugs"):
        return False, "No recognized drugs in the question."

    # Basic check: no evidence at all
    total_claims = 0
    total_passages = 0
    for drug_name, drug_data in evidence_package.get("drugs", {}).items():
        nbn = drug_data.get("nbn_evidence", [])
        stahl = drug_data.get("stahl_evidence", [])
        total_claims += len(nbn) + len(stahl)
    total_passages = len(evidence_package.get("kaplan_passages", []))

    if total_claims == 0 and total_passages == 0:
        return False, "No evidence found for the requested drug(s)."

    # If no understanding provided, basic check is sufficient
    if understanding is None:
        return True, None

    question_type = understanding.get("question_type", "general")
    is_patient_specific = understanding.get("is_patient_specific", False)
    premises = understanding.get("premises", [])

    # Patient-specific questions always need clarification
    if is_patient_specific:
        return False, (
            "This question appears to be about a specific patient. "
            "The system cannot provide patient-specific recommendations. "
            "Please consult a qualified clinician or rephrase as a general question."
        )

    # False-premise detection: if premises are stated, they should be checkable
    # against evidence. For now, we flag premises for the LLM to verify but
    # don't abstain solely on premises — the entailment gate will catch
    # unsupported premise-based claims.

    # Question-type-specific sufficiency
    if question_type in REQUIRED_CATEGORIES:
        return _check_type_specific_sufficiency(evidence_package, question_type)

    # Comparison questions: need evidence for all drugs
    if question_type == "comparison":
        drug_names = understanding.get("drug_names", [])
        if len(drug_names) < 2:
            return False, "Comparison questions require at least two drugs to compare."
        drugs_with_evidence = 0
        for drug_name, drug_data in evidence_package.get("drugs", {}).items():
            nbn = drug_data.get("nbn_evidence", [])
            stahl = drug_data.get("stahl_evidence", [])
            if nbn or stahl:
                drugs_with_evidence += 1
        if drugs_with_evidence < len(drug_names):
            return False, (
                f"Comparison requires evidence for all {len(drug_names)} drugs, "
                f"but only {drugs_with_evidence} have evidence."
            )

    return True, None


def _check_type_specific_sufficiency(
    evidence_package: dict[str, Any],
    question_type: str,
) -> tuple[bool, str | None]:
    """Check question-type-specific sufficiency rules."""
    required_alternatives = REQUIRED_CATEGORIES[question_type]

    # Collect all categories present in the evidence
    present_categories: set[str] = set()
    for drug_name, drug_data in evidence_package.get("drugs", {}).items():
        for e in drug_data.get("nbn_evidence", []):
            present_categories.add(e.get("category", ""))
        for e in drug_data.get("stahl_evidence", []):
            present_categories.add(e.get("category", ""))

    # Check if any alternative set is fully present
    for required_set in required_alternatives:
        if required_set.issubset(present_categories):
            return True, None

    # Not sufficient — explain what's missing
    missing_desc = " or ".join(
        ", ".join(sorted(s)) for s in required_alternatives
    )
    return False, (
        f"Insufficient evidence for a {question_type} question. "
        f"Required categories: {missing_desc}. "
        f"Available: {sorted(present_categories)}"
    )
