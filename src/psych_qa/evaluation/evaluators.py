"""Evaluators — check answers against gold case expectations.

Each evaluator returns a dict with: passed, score, details.
"""

from __future__ import annotations

from typing import Any

from ..answering.entailment import judge_claims_batch
from .matchers import compute_recall_at_k, build_drug_slug_lookup


def evaluate_status(
    answer: dict[str, Any],
    expected_status: str,
) -> dict[str, Any]:
    """Check if the answer status matches the expected status."""
    actual = answer.get("status", "answered")
    passed = actual == expected_status
    return {
        "evaluator": "status",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "details": {"expected": expected_status, "actual": actual},
    }


def evaluate_recall(
    required_points: list[Any],
    evidence_package: dict[str, Any],
) -> dict[str, Any]:
    """Compute Recall@K for required answer points."""
    drug_slug_lookup = build_drug_slug_lookup()
    result = compute_recall_at_k(required_points, evidence_package, drug_slug_lookup)
    passed = result["recall"] >= 0.9
    return {
        "evaluator": "recall_at_k",
        "passed": passed,
        "score": result["recall"],
        "details": {
            "points_covered": result["points_covered"],
            "points_total": result["points_total"],
            "per_point": result["per_point"],
        },
    }


def evaluate_required_points(
    answer: dict[str, Any],
    required_points: list[Any],
) -> dict[str, Any]:
    """Check if required answer points are covered in the answer claims.

    Uses text matching (case-insensitive substring) as a simple check.
    A more sophisticated version would use an LLM judge.
    """
    claims = answer.get("claims", [])
    claim_texts = [c.get("text", "").lower() for c in claims]

    covered = 0
    per_point = []
    for point in required_points:
        point_text = point.text.lower()
        # Check if any claim contains the point text (simplified)
        found = any(
            _text_overlap(point_text, claim_text)
            for claim_text in claim_texts
        )
        if found:
            covered += 1
        per_point.append({"point_text": point.text, "found": found})

    score = covered / len(required_points) if required_points else 1.0
    passed = score >= 0.9
    return {
        "evaluator": "required_points",
        "passed": passed,
        "score": score,
        "details": {"covered": covered, "total": len(required_points), "per_point": per_point},
    }


def _text_overlap(expected: str, actual: str) -> bool:
    """Check if key terms from expected text appear in actual text."""
    # Extract key terms (words > 4 chars)
    expected_terms = {w for w in expected.split() if len(w) > 4}
    if not expected_terms:
        return expected in actual

    actual_terms = set(actual.split())
    overlap = expected_terms & actual_terms
    # At least 50% of expected terms should be present
    return len(overlap) / len(expected_terms) >= 0.5


def evaluate_forbidden_claims(
    answer: dict[str, Any],
    forbidden_claims: list[str],
) -> dict[str, Any]:
    """Check that no forbidden claims appear in the answer."""
    claims = answer.get("claims", [])
    claim_texts = [c.get("text", "").lower() for c in claims]

    violations = []
    for forbidden in forbidden_claims:
        forbidden_lower = forbidden.lower()
        for claim_text in claim_texts:
            if _text_overlap(forbidden_lower, claim_text):
                violations.append({"forbidden": forbidden, "claim_text": claim_text})

    passed = len(violations) == 0
    return {
        "evaluator": "forbidden_claims",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "details": {"violations": violations, "total_forbidden": len(forbidden_claims)},
    }


def evaluate_citation_validity(
    answer: dict[str, Any],
    evidence_package: dict[str, Any],
) -> dict[str, Any]:
    """Check that all cited evidence IDs exist in the evidence package."""
    from ..answering.citations import validate_citations

    valid_ids = evidence_package.get("required_citations", [])
    is_valid, invalid_ids = validate_citations(answer, valid_ids)
    return {
        "evaluator": "citation_validity",
        "passed": is_valid,
        "score": 1.0 if is_valid else 0.0,
        "details": {"invalid_ids": invalid_ids},
    }


def evaluate_entailment(
    answer: dict[str, Any],
    evidence_package: dict[str, Any],
) -> dict[str, Any]:
    """Judge claim-to-evidence entailment for all claims."""
    from ..answering.answer_service import build_evidence_lookup

    claims = answer.get("claims", [])
    if not claims:
        return {
            "evaluator": "entailment",
            "passed": True,
            "score": 1.0,
            "details": {"verdicts": [], "note": "No claims to evaluate"},
        }

    evidence_lookup = build_evidence_lookup(evidence_package)
    verdicts = judge_claims_batch(claims, evidence_lookup)

    supported = sum(1 for v in verdicts if v["verdict"] == "SUPPORTED")
    critical_verdicts = [v for v in verdicts if v.get("is_critical")]
    critical_supported = sum(1 for v in critical_verdicts if v["verdict"] == "SUPPORTED")

    score = supported / len(claims) if claims else 1.0
    critical_score = critical_supported / len(critical_verdicts) if critical_verdicts else 1.0

    # Hard gate: all critical claims must be SUPPORTED
    passed = critical_score == 1.0

    return {
        "evaluator": "entailment",
        "passed": passed,
        "score": score,
        "details": {
            "verdicts": verdicts,
            "supported": supported,
            "total": len(claims),
            "critical_supported": critical_supported,
            "critical_total": len(critical_verdicts),
            "critical_score": critical_score,
        },
    }
