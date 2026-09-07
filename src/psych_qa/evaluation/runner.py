"""Evaluation runner — runs the full evaluation suite and produces a release report.

Supports:
- Dataset loading (gold, abstention, regression)
- Answer execution via answer_service
- Retrieval Recall@K against gold evidence
- Required point checks
- Forbidden claim checks
- Status/abstention checks
- Citation validity + entailment
- Repeated stochastic runs
- Release-gate reporting
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..answering.answer_service import answer_question
from ..config import get_settings
from .evaluators import (
    evaluate_citation_validity,
    evaluate_entailment,
    evaluate_forbidden_claims,
    evaluate_recall,
    evaluate_required_points,
    evaluate_status,
)
from .loader import has_unapproved_cases, load_dataset
from .schemas import GoldCase, AbstentionCase, RegressionCase

logger = logging.getLogger(__name__)


# Release gates for Amisulpride
RELEASE_GATES = {
    "critical_source_extraction_accuracy": {"gate": 1.0, "description": "100% critical source extraction"},
    "retrieval_recall_at_5": {"gate": 0.9, "description": "≥90% Recall@5 on gold evidence"},
    "unsupported_critical_claims": {"gate": 0, "description": "0 unsupported critical claims"},
    "citation_entailment_critical": {"gate": 1.0, "description": "100% citation entailment for critical claims"},
    "citation_completeness": {"gate": 0.95, "description": "≥95% overall citation completeness"},
    "correct_abstention": {"gate": 0.9, "description": "≥90% correct abstention on unsupported questions"},
    "psychiatrist_approved": {"gate": 0.9, "description": "≥90% psychiatrist-approved answers"},
    "critical_clinical_errors": {"gate": 0, "description": "0 critical clinical errors"},
}


def run_single_case(question: str) -> dict[str, Any]:
    """Run a single question through the answer pipeline."""
    result = answer_question(question)
    return result


def evaluate_gold_case(
    case: GoldCase,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a gold case against the answer result."""
    answer = result["answer"]
    evidence_package = result["evidence_package"]

    evaluations = {
        "status": evaluate_status(answer, case.expected_status.value),
        "recall": evaluate_recall(case.required_answer_points, evidence_package),
        "required_points": evaluate_required_points(answer, case.required_answer_points),
        "forbidden_claims": evaluate_forbidden_claims(answer, case.forbidden_claims),
        "citation_validity": evaluate_citation_validity(answer, evidence_package),
        "entailment": evaluate_entailment(answer, evidence_package),
    }

    all_passed = all(e["passed"] for e in evaluations.values())
    return {
        "question": case.question,
        "expected_status": case.expected_status.value,
        "actual_status": answer.get("status"),
        "evaluations": evaluations,
        "all_passed": all_passed,
        "metrics": result.get("metrics", {}),
    }


def evaluate_abstention_case(
    case: AbstentionCase,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate an abstention case — checks that the system abstained correctly."""
    answer = result["answer"]

    status_eval = evaluate_status(answer, case.expected_status.value)

    return {
        "question": case.question,
        "expected_status": case.expected_status.value,
        "actual_status": answer.get("status"),
        "reason": case.reason,
        "evaluations": {"status": status_eval},
        "all_passed": status_eval["passed"],
        "metrics": result.get("metrics", {}),
    }


def run_dataset(
    dataset: str,
    repeats: int = 1,
) -> dict[str, Any]:
    """Run evaluation on a dataset with optional repeated runs.

    Args:
        dataset: Dataset name (golden_v1, abstention_v1, regression).
        repeats: Number of times to run each case (for stochastic variance).

    Returns:
        Dict with per-case results and aggregate metrics.
    """
    cases = load_dataset(dataset)
    logger.info(f"Loaded {len(cases)} cases from {dataset}")

    if has_unapproved_cases(cases):
        logger.warning(
            f"Dataset {dataset} contains cases with review_status='needs_sasson_approval'. "
            "Psychiatrist-approved metrics will not be computed."
        )

    all_results: list[dict[str, Any]] = []
    for i, case in enumerate(cases):
        logger.info(f"Running case {i+1}/{len(cases)}: {case.question[:60]}...")

        repeat_results = []
        for r in range(repeats):
            result = run_single_case(case.question)

            if isinstance(case, (GoldCase, RegressionCase)):
                eval_result = evaluate_gold_case(case, result)
            elif isinstance(case, AbstentionCase):
                eval_result = evaluate_abstention_case(case, result)
            else:
                eval_result = evaluate_gold_case(case, result)

            eval_result["repeat"] = r
            repeat_results.append(eval_result)

        # Aggregate across repeats
        if repeats > 1:
            pass_rates = [r["all_passed"] for r in repeat_results]
            avg_pass = sum(pass_rates) / len(pass_rates)
            all_results.append({
                "question": case.question,
                "repeats": repeat_results,
                "avg_pass_rate": avg_pass,
            })
        else:
            all_results.append(repeat_results[0])

    return {
        "dataset": dataset,
        "total_cases": len(cases),
        "repeats": repeats,
        "results": all_results,
        "timestamp": datetime.utcnow().isoformat(),
    }


def compute_release_report(
    gold_results: dict[str, Any],
    abstention_results: dict[str, Any],
    regression_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the release report against the release gates."""
    # Gold metrics
    gold_passes = [r["all_passed"] for r in gold_results["results"]]
    gold_pass_rate = sum(gold_passes) / len(gold_passes) if gold_passes else 0

    # Recall@5
    recall_scores = []
    for r in gold_results["results"]:
        recall = r.get("evaluations", {}).get("recall", {})
        if "score" in recall:
            recall_scores.append(recall["score"])
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0

    # Citation completeness
    completeness_scores = []
    for r in gold_results["results"]:
        m = r.get("metrics", {})
        if "citation_completeness" in m:
            completeness_scores.append(m["citation_completeness"])
    avg_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0

    # Unsupported critical claims
    total_unsupported_critical = 0
    for r in gold_results["results"]:
        m = r.get("metrics", {})
        total_unsupported_critical += m.get("unsupported_critical_claims", 0)

    # Entailment for critical claims
    critical_entailment_scores = []
    for r in gold_results["results"]:
        entail = r.get("evaluations", {}).get("entailment", {})
        if "details" in entail and "critical_score" in entail["details"]:
            critical_entailment_scores.append(entail["details"]["critical_score"])
    avg_critical_entailment = (
        sum(critical_entailment_scores) / len(critical_entailment_scores)
        if critical_entailment_scores else 0
    )

    # Abstention rate
    abstention_passes = [r["all_passed"] for r in abstention_results["results"]]
    abstention_pass_rate = sum(abstention_passes) / len(abstention_passes) if abstention_passes else 0

    # Psychiatrist approval (from case review_status — not yet available)
    # This will be computed once clinician reviews are stored
    psychiatrist_approved = None  # Cannot compute yet

    # Critical clinical errors (from clinician reviews — not yet available)
    critical_clinical_errors = None  # Cannot compute yet

    gates = {
        "retrieval_recall_at_5": {
            "value": avg_recall,
            "gate": RELEASE_GATES["retrieval_recall_at_5"]["gate"],
            "passed": avg_recall >= RELEASE_GATES["retrieval_recall_at_5"]["gate"],
            "description": RELEASE_GATES["retrieval_recall_at_5"]["description"],
        },
        "unsupported_critical_claims": {
            "value": total_unsupported_critical,
            "gate": RELEASE_GATES["unsupported_critical_claims"]["gate"],
            "passed": total_unsupported_critical == RELEASE_GATES["unsupported_critical_claims"]["gate"],
            "description": RELEASE_GATES["unsupported_critical_claims"]["description"],
        },
        "citation_entailment_critical": {
            "value": avg_critical_entailment,
            "gate": RELEASE_GATES["citation_entailment_critical"]["gate"],
            "passed": avg_critical_entailment >= RELEASE_GATES["citation_entailment_critical"]["gate"],
            "description": RELEASE_GATES["citation_entailment_critical"]["description"],
        },
        "citation_completeness": {
            "value": avg_completeness,
            "gate": RELEASE_GATES["citation_completeness"]["gate"],
            "passed": avg_completeness >= RELEASE_GATES["citation_completeness"]["gate"],
            "description": RELEASE_GATES["citation_completeness"]["description"],
        },
        "correct_abstention": {
            "value": abstention_pass_rate,
            "gate": RELEASE_GATES["correct_abstention"]["gate"],
            "passed": abstention_pass_rate >= RELEASE_GATES["correct_abstention"]["gate"],
            "description": RELEASE_GATES["correct_abstention"]["description"],
        },
        "psychiatrist_approved": {
            "value": psychiatrist_approved,
            "gate": RELEASE_GATES["psychiatrist_approved"]["gate"],
            "passed": None,  # Cannot compute yet
            "description": RELEASE_GATES["psychiatrist_approved"]["description"],
        },
        "critical_clinical_errors": {
            "value": critical_clinical_errors,
            "gate": RELEASE_GATES["critical_clinical_errors"]["gate"],
            "passed": None,  # Cannot compute yet
            "description": RELEASE_GATES["critical_clinical_errors"]["description"],
        },
    }

    all_computable_gates_passed = all(
        g["passed"] for g in gates.values() if g["passed"] is not None
    )

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "gold_pass_rate": gold_pass_rate,
        "abstention_pass_rate": abstention_pass_rate,
        "gates": gates,
        "all_computable_gates_passed": all_computable_gates_passed,
        "ready_for_release": all_computable_gates_passed and psychiatrist_approved is not None,
        "notes": [
            "Psychiatrist approval and critical clinical errors require clinician review.",
            "Run multiple times to account for generative variance.",
        ],
    }


def save_report(report: dict[str, Any], path: Path | None = None) -> Path:
    """Save a release report to a JSON file."""
    if path is None:
        settings = get_settings()
        path = settings.evals_dir / f"release_report_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return path
