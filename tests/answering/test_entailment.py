"""Tests for entailment validation logic (without LLM calls)."""

from __future__ import annotations

from unittest.mock import patch

from psych_qa.answering.entailment import (
    judge_claim,
    judge_claims_batch,
    should_abstain_after_regeneration,
    should_regenerate,
)


class TestShouldRegenerate:
    def test_regenerate_on_unsupported_critical(self):
        verdicts = [
            {"is_critical": True, "verdict": "UNSUPPORTED", "claim_text": "Dose is 50mg"},
            {"is_critical": False, "verdict": "SUPPORTED", "claim_text": "Mechanism"},
        ]
        regen, reasons = should_regenerate(verdicts)
        assert regen is True
        assert len(reasons) == 1

    def test_no_regenerate_when_all_supported(self):
        verdicts = [
            {"is_critical": True, "verdict": "SUPPORTED", "claim_text": "Dose is 50mg"},
            {"is_critical": False, "verdict": "SUPPORTED", "claim_text": "Mechanism"},
        ]
        regen, reasons = should_regenerate(verdicts)
        assert regen is False
        assert reasons == []

    def test_no_regenerate_for_noncritical_unsupported(self):
        """Non-critical unsupported claims don't trigger regeneration."""
        verdicts = [
            {"is_critical": False, "verdict": "UNSUPPORTED", "claim_text": "Minor detail"},
        ]
        regen, _ = should_regenerate(verdicts)
        assert regen is False

    def test_regenerate_on_contradicted_critical(self):
        verdicts = [
            {"is_critical": True, "verdict": "CONTRADICTED", "claim_text": "Dose is 50mg"},
        ]
        regen, reasons = should_regenerate(verdicts)
        assert regen is True

    def test_regenerate_on_unsupported_specificity_critical(self):
        verdicts = [
            {"is_critical": True, "verdict": "UNSUPPORTED_SPECIFICITY", "claim_text": "Dose"},
        ]
        regen, _ = should_regenerate(verdicts)
        assert regen is True


class TestShouldAbstain:
    def test_abstain_after_failed_regeneration(self):
        verdicts = [
            {"is_critical": True, "verdict": "UNSUPPORTED", "claim_text": "Dose"},
        ]
        abstain, _ = should_abstain_after_regeneration(verdicts)
        assert abstain is True

    def test_no_abstain_if_all_critical_supported(self):
        verdicts = [
            {"is_critical": True, "verdict": "SUPPORTED", "claim_text": "Dose"},
        ]
        abstain, _ = should_abstain_after_regeneration(verdicts)
        assert abstain is False


class TestJudgeClaim:
    def test_missing_citation(self):
        """Claim with no evidence should return MISSING_CITATION."""
        result = judge_claim("Some claim", evidence_texts=[], is_critical=True)
        assert result["verdict"] == "MISSING_CITATION"

    def test_judge_with_mocked_llm(self):
        """Test that judge_claim calls the LLM and returns its verdict."""
        mock_result = {
            "verdict": "SUPPORTED",
            "reasoning": "Evidence supports the claim.",
            "unsupported_specificity_detail": None,
        }
        with patch("psych_qa.answering.entailment.get_llm_client") as mock_client:
            mock_client.return_value.chat_structured.return_value = (mock_result, 0, 0, 0)
            result = judge_claim("Dose is 400mg", ["The dose is 400-800mg/day"], is_critical=True)
        assert result["verdict"] == "SUPPORTED"
        assert result["reasoning"] == "Evidence supports the claim."


class TestJudgeClaimsBatch:
    def test_batch_with_mocked_llm(self):
        claims = [
            {"text": "Dose is 400mg", "evidence_ids": ["stahl_claim_1"], "claim_role": "direct", "critical": True},
            {"text": "Mechanism is D2 blockade", "evidence_ids": ["stahl_claim_2"], "claim_role": "direct", "critical": False},
        ]
        evidence_lookup = {
            "stahl_claim_1": {"text": "Dose is 400-800mg/day"},
            "stahl_claim_2": {"text": "Blocks D2 receptors"},
        }
        mock_result = {
            "verdict": "SUPPORTED",
            "reasoning": "Supported.",
            "unsupported_specificity_detail": None,
        }
        with patch("psych_qa.answering.entailment.get_llm_client") as mock_client:
            mock_client.return_value.chat_structured.return_value = (mock_result, 0, 0, 0)
            verdicts = judge_claims_batch(claims, evidence_lookup)

        assert len(verdicts) == 2
        assert verdicts[0]["verdict"] == "SUPPORTED"
        assert verdicts[0]["is_critical"] is True
        assert verdicts[1]["is_critical"] is False

    def test_batch_missing_evidence(self):
        """Claims with evidence_ids not in lookup get MISSING_CITATION."""
        claims = [
            {"text": "Some claim", "evidence_ids": ["nonexistent"], "claim_role": "direct", "critical": False},
        ]
        verdicts = judge_claims_batch(claims, {})
        assert verdicts[0]["verdict"] == "MISSING_CITATION"
