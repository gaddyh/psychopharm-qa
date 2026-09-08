"""Tests for evaluation evaluators (without LLM calls)."""

from __future__ import annotations

from psych_qa.evaluation.evaluators import (
    evaluate_citation_validity,
    evaluate_forbidden_claims,
    evaluate_recall,
    evaluate_required_points,
    evaluate_status,
)
from psych_qa.evaluation.schemas import AcceptableEvidence, RequiredAnswerPoint


class TestEvaluateStatus:
    def test_correct_status(self):
        result = evaluate_status({"status": "answered"}, "answered")
        assert result["passed"] is True
        assert result["score"] == 1.0

    def test_wrong_status(self):
        result = evaluate_status({"status": "abstained"}, "answered")
        assert result["passed"] is False
        assert result["score"] == 0.0


class TestEvaluateForbiddenClaims:
    def test_no_violations(self):
        answer = {"claims": [
            {"text": "Dose is 400mg"},
            {"text": "Mechanism is D2 blockade"},
        ]}
        forbidden = ["amisulpride is a typical antipsychotic"]
        result = evaluate_forbidden_claims(answer, forbidden)
        assert result["passed"] is True
        assert result["details"]["violations"] == []

    def test_violation_found(self):
        answer = {"claims": [
            {"text": "amisulpride is a typical antipsychotic drug"},
        ]}
        forbidden = ["amisulpride is a typical antipsychotic"]
        result = evaluate_forbidden_claims(answer, forbidden)
        assert result["passed"] is False
        assert len(result["details"]["violations"]) > 0


class TestEvaluateCitationValidity:
    def test_all_valid(self):
        answer = {"claims": [
            {"text": "a", "evidence_ids": ["stahl_claim_1"]},
        ]}
        ep = {"required_citations": ["stahl_claim_1"]}
        result = evaluate_citation_validity(answer, ep)
        assert result["passed"] is True

    def test_invalid_id(self):
        answer = {"claims": [
            {"text": "a", "evidence_ids": ["fake_id"]},
        ]}
        ep = {"required_citations": ["stahl_claim_1"]}
        result = evaluate_citation_validity(answer, ep)
        assert result["passed"] is False
        assert "fake_id" in result["details"]["invalid_ids"]


class TestEvaluateRequiredPoints:
    def test_all_points_covered(self):
        answer = {"claims": [
            {"text": "The dose is 400-800 mg/day in 2 doses for schizophrenia"},
            {"text": "Initial dosing should be 400-800 mg/day"},
        ]}
        points = [
            RequiredAnswerPoint(text="Schizophrenia dose is 400-800 mg/day"),
            RequiredAnswerPoint(text="Initial dosing is 400-800 mg/day"),
        ]
        result = evaluate_required_points(answer, points)
        assert result["score"] >= 0.9

    def test_no_points_covered(self):
        answer = {"claims": [
            {"text": "Something completely unrelated"},
        ]}
        points = [
            RequiredAnswerPoint(text="Schizophrenia dose is 400-800 mg/day"),
        ]
        result = evaluate_required_points(answer, points)
        assert result["score"] < 0.5

    def test_empty_points(self):
        result = evaluate_required_points({"claims": []}, [])
        assert result["score"] == 1.0


class TestEvaluateRecall:
    def test_recall_with_mocked_lookup(self, monkeypatch):
        """Test recall computation with mocked drug slug lookup."""
        def mock_lookup():
            return {22: "amisulpride"}
        monkeypatch.setattr(
            "psych_qa.evaluation.evaluators.build_drug_slug_lookup",
            mock_lookup,
        )

        ep = {
            "drugs": {
                "amisulpride": {
                    "nbn_evidence": [],
                    "stahl_evidence": [
                        {
                            "evidence_id": "stahl_claim_1",
                            "source": "stahl",
                            "category": "mechanism",
                            "text": "Blocks D2 receptors",
                            "attributes": {"claim_hash": "abc123"},
                            "locator": {},
                        },
                    ],
                },
            },
            "kaplan_passages": [],
        }
        points = [
            RequiredAnswerPoint(
                text="D2 blockade",
                acceptable_evidence=[
                    AcceptableEvidence(source="stahl", category="mechanism", claim_hash="abc123"),
                ],
            ),
        ]
        result = evaluate_recall(points, ep)
        assert result["score"] == 1.0
        assert result["passed"] is True
