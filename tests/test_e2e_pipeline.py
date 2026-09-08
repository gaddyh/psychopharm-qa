"""End-to-end integration tests for the full answer pipeline.

These tests mock the LLM client to avoid API calls but exercise the full
pipeline: question parsing → drug lookup → evidence building → answer
generation → criticality → citation validation → entailment → prose assembly.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture
def mock_llm_client():
    """Mock the LLM client to return predictable responses."""
    mock_client = patch("psych_qa.answering.answer_service.get_llm_client")
    mock_parser = patch("psych_qa.retrieval.question_parser.get_llm_client")
    mock_reranker = patch("psych_qa.retrieval.reranker.get_llm_client")
    mock_entailment = patch("psych_qa.answering.entailment.get_llm_client")

    # Question parsing response
    parse_response = {
        "drug_names": ["amisulpride"],
        "question_type": "mechanism",
        "concepts": ["mechanism", "dopamine"],
        "premises": [],
        "is_patient_specific": False,
    }

    # Answer generation response (claims-first V2)
    answer_response = {
        "claims": [
            {
                "text": "Amisulpride blocks presynaptic dopamine D2 receptors at low doses",
                "evidence_ids": ["stahl_claim_2107"],
                "claim_role": "direct",
            },
            {
                "text": "It blocks postsynaptic D2 receptors at higher doses",
                "evidence_ids": ["stahl_claim_2108"],
                "claim_role": "direct",
            },
        ],
        "status": "answered",
        "uncertainties": [],
        "clarification_question": None,
    }

    # Reranker response
    reranker_response = {"scores": [9.0, 8.0, 7.0, 6.0, 5.0]}

    # Entailment response
    entailment_response = {
        "verdict": "SUPPORTED",
        "reasoning": "Evidence supports the claim.",
        "unsupported_specificity_detail": None,
    }

    # Set up mocks
    client_mock = mock_client.start()
    parser_mock = mock_parser.start()
    reranker_mock = mock_reranker.start()
    entailment_mock = mock_entailment.start()

    # Configure the mock client
    fake_client = client_mock.return_value
    fake_client.chat_model = "test-model"
    fake_client.chat_structured.return_value = (answer_response, 100, 50, 500)
    fake_client.chat_structured_timed.return_value = (answer_response, 100, 50, 500)

    # Configure parser mock
    parser_mock.return_value.chat_structured.return_value = (parse_response, 50, 20, 100)

    # Configure reranker mock
    reranker_mock.return_value.chat_structured.return_value = (reranker_response, 50, 20, 100)

    # Configure entailment mock
    entailment_mock.return_value.chat_structured.return_value = (entailment_response, 50, 20, 100)

    yield fake_client

    mock_client.stop()
    mock_parser.stop()
    mock_reranker.stop()
    mock_entailment.stop()


class TestFullPipelineMechanism:
    """Integration test: full pipeline for a mechanism question."""

    def test_mechanism_question_produces_claims(self, mock_llm_client):
        """A mechanism question should produce claims with citations."""
        from psych_qa.answering.answer_service import answer_question

        result = answer_question("What is the mechanism of action of amisulpride?")

        # Should have an answer with claims
        answer = result["answer"]
        assert answer["status"] == "answered"
        assert len(answer["claims"]) > 0

        # Every claim should have at least one evidence_id
        for claim in answer["claims"]:
            assert len(claim["evidence_ids"]) > 0, f"Claim '{claim['text'][:40]}...' has no evidence"

        # Should have assembled prose
        assert answer["direct_answer"] != ""

        # Should have metrics
        metrics = result["metrics"]
        assert "total_claims" in metrics
        assert metrics["total_claims"] > 0
        assert "citation_completeness" in metrics

        # Should have version metadata
        versions = result["versions"]
        assert versions["answer_schema_version"] == 2
        assert versions["prompt_version"] == "claims-v1"
        assert versions["entailment_prompt_version"] == "entailment-v1"

    def test_trace_saved_to_db(self, mock_llm_client):
        """The answer trace should be saved to the database."""
        from psych_qa.answering.answer_service import answer_question

        result = answer_question("What is the mechanism of amisulpride?")
        assert result["trace_id"] is not None
        assert result["trace_id"] > 0


class TestFullPipelineAbstention:
    """Integration test: pipeline should abstain on patient-specific questions."""

    def test_patient_specific_abstains(self, mock_llm_client):
        """A patient-specific question should trigger abstention."""
        # Override the parse response for this test
        from psych_qa.answering.answer_service import answer_question

        with patch("psych_qa.retrieval.question_parser.get_llm_client") as mock_parser:
            mock_parser.return_value.chat_structured.return_value = (
                {
                    "drug_names": ["amisulpride"],
                    "question_type": "dosing",
                    "concepts": ["dose"],
                    "premises": [],
                    "is_patient_specific": True,
                },
                50, 20, 100,
            )
            result = answer_question("Should I give my patient 800mg of amisulpride?")

        answer = result["answer"]
        assert answer["status"] == "abstained"
        assert len(answer["claims"]) == 0


class TestFullPipelineEvidenceFiltering:
    """Integration test: evidence should be filtered by question type."""

    def test_dosing_question_gets_dosing_evidence(self, mock_llm_client):
        """A dosing question should retrieve dosing-related evidence."""
        from psych_qa.answering.answer_service import answer_question

        with patch("psych_qa.retrieval.question_parser.get_llm_client") as mock_parser:
            mock_parser.return_value.chat_structured.return_value = (
                {
                    "drug_names": ["amisulpride"],
                    "question_type": "dosing",
                    "concepts": ["dose"],
                    "premises": [],
                    "is_patient_specific": False,
                },
                50, 20, 100,
            )
            result = answer_question("What is the dose of amisulpride?")

        ep = result["evidence_package"]
        # Should have Stahl evidence with dosing categories
        for drug_name, drug_data in ep.get("drugs", {}).items():
            stahl = drug_data.get("stahl_evidence", [])
            if stahl:
                categories = {e.get("category") for e in stahl}
                # Dosing questions should include dosing categories
                assert any(cat in categories for cat in ["dosage_range", "dosing_instructions", "dosing"]), (
                    f"Dosing question should retrieve dosing evidence, got: {categories}"
                )
                # Should NOT include unrelated categories like "pearls"
                assert "pearls" not in categories, "Dosing question should not retrieve pearls"


class TestDatasetIntegrity:
    """Test that the gold and abstention datasets are valid."""

    def test_gold_dataset_loads(self):
        from psych_qa.evaluation.loader import load_dataset
        cases = load_dataset("golden_v1")
        assert len(cases) >= 15
        for case in cases:
            assert case.question
            assert case.expected_status
            assert case.review_status is not None

    def test_abstention_dataset_loads(self):
        from psych_qa.evaluation.loader import load_dataset
        cases = load_dataset("abstention_v1")
        assert len(cases) >= 10
        for case in cases:
            assert case.question
            assert case.expected_status in ["abstained", "needs_clarification"]

    def test_all_cases_need_approval(self):
        """All cases should start as needs_sasson_approval."""
        from psych_qa.evaluation.loader import load_dataset
        from psych_qa.evaluation.schemas import ReviewStatus

        gold = load_dataset("golden_v1")
        abst = load_dataset("abstention_v1")
        for case in gold + abst:
            assert case.review_status == ReviewStatus.NEEDS_SASSON_APPROVAL, (
                f"Case '{case.question[:40]}...' should need approval"
            )
