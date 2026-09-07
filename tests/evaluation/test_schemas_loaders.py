"""Tests for evaluation schemas, loader, and matchers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from psych_qa.evaluation.schemas import (
    AbstentionCase,
    AcceptableEvidence,
    GoldCase,
    RegressionCase,
    ReviewStatus,
    TraceVersions,
    parse_case,
)
from psych_qa.evaluation.matchers import (
    compute_recall_at_k,
    match_evidence_item,
    resolve_acceptable_evidence,
)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_gold_case_defaults(self):
        case = GoldCase(question="What is the mechanism of amisulpride?")
        assert case.expected_status.value == "answered"
        assert case.review_status == ReviewStatus.NEEDS_SASSON_APPROVAL
        assert case.required_answer_points == []
        assert case.forbidden_claims == []

    def test_abstention_case(self):
        case = AbstentionCase(
            question="Should I give amisulpride to my pregnant patient?",
            expected_status="needs_clarification",
        )
        assert case.expected_status.value == "needs_clarification"

    def test_regression_case(self):
        case = RegressionCase(
            question="What is the dose of amisulpride?",
            source_trace_id=42,
        )
        assert case.source_trace_id == 42
        assert case.review_status == ReviewStatus.NEEDS_SASSON_APPROVAL

    def test_parse_case_gold(self):
        raw = {"question": "test?", "expected_status": "answered"}
        case = parse_case(raw, "golden_v1")
        assert isinstance(case, GoldCase)

    def test_parse_case_abstention(self):
        raw = {"question": "test?", "expected_status": "abstained"}
        case = parse_case(raw, "abstention_v1")
        assert isinstance(case, AbstentionCase)

    def test_parse_case_regression(self):
        raw = {"question": "test?", "expected_status": "answered"}
        case = parse_case(raw, "regression")
        assert isinstance(case, RegressionCase)

    def test_trace_versions_defaults(self):
        v = TraceVersions()
        assert v.answer_schema_version == 2
        assert v.prompt_version == "claims-v1"
        assert v.retriever_version == "hybrid-v1"

    def test_acceptable_evidence_with_claim_hash(self):
        ae = AcceptableEvidence(
            source="stahl",
            drug_slug="amisulpride",
            category="mechanism",
            claim_hash="abc123",
        )
        assert ae.claim_hash == "abc123"
        assert ae.source == "stahl"

    def test_acceptable_evidence_with_locator(self):
        ae = AcceptableEvidence(
            source="kaplan",
            locator={"physical_pdf_page": 10050},
        )
        assert ae.locator == {"physical_pdf_page": 10050}


# ---------------------------------------------------------------------------
# Matcher tests
# ---------------------------------------------------------------------------

class TestMatchers:
    @pytest.fixture
    def evidence_package(self):
        """A synthetic evidence package for testing."""
        return {
            "drugs": {
                "amisulpride": {
                    "nbn_evidence": [
                        {
                            "evidence_id": "nbn_claim_1",
                            "source": "nbn",
                            "drug_id": 22,
                            "category": "mode_of_action",
                            "text": "MoA Pre-synaptic antagonist",
                            "attributes": {"claim_hash": "hash_a", "dose_band": "low"},
                            "locator": {"nbn_id": 167},
                        },
                    ],
                    "stahl_evidence": [
                        {
                            "evidence_id": "stahl_claim_100",
                            "source": "stahl",
                            "drug_id": 22,
                            "category": "mechanism",
                            "text": "Blocks dopamine D2 receptors presynaptically",
                            "attributes": {"claim_hash": "hash_b", "fda_approved": False},
                            "locator": {"printed_book_page": 17, "physical_pdf_page": 33},
                        },
                    ],
                },
            },
            "kaplan_passages": [
                {
                    "evidence_id": "kaplan_chunk_500",
                    "source": "kaplan",
                    "text": "Amisulpride is a benzamide antipsychotic that blocks D2 receptors.",
                    "attributes": {"score": 0.9},
                    "locator": {"physical_pdf_page": 10050},
                },
            ],
        }

    def test_match_by_claim_hash(self, evidence_package):
        ae = AcceptableEvidence(
            source="stahl",
            drug_slug="amisulpride",
            category="mechanism",
            claim_hash="hash_b",
        )
        result = resolve_acceptable_evidence(ae, evidence_package, {22: "amisulpride"})
        assert "stahl_claim_100" in result

    def test_match_by_locator_kaplan(self, evidence_package):
        ae = AcceptableEvidence(
            source="kaplan",
            locator={"physical_pdf_page": 10050},
        )
        result = resolve_acceptable_evidence(ae, evidence_package)
        assert "kaplan_chunk_500" in result

    def test_match_by_text_contains(self, evidence_package):
        ae = AcceptableEvidence(
            source="stahl",
            category="mechanism",
            text_contains="presynaptically",
        )
        result = resolve_acceptable_evidence(ae, evidence_package)
        assert "stahl_claim_100" in result

    def test_no_match_wrong_source(self, evidence_package):
        ae = AcceptableEvidence(
            source="nbn",
            category="mechanism",
            claim_hash="hash_b",
        )
        result = resolve_acceptable_evidence(ae, evidence_package)
        assert result == []

    def test_compute_recall_all_found(self, evidence_package):
        from psych_qa.evaluation.schemas import RequiredAnswerPoint
        points = [
            RequiredAnswerPoint(
                text="Mechanism involves D2 blockade",
                acceptable_evidence=[
                    AcceptableEvidence(source="stahl", category="mechanism", claim_hash="hash_b"),
                ],
            ),
            RequiredAnswerPoint(
                text="Kaplan discusses amisulpride",
                acceptable_evidence=[
                    AcceptableEvidence(source="kaplan", locator={"physical_pdf_page": 10050}),
                ],
            ),
        ]
        result = compute_recall_at_k(points, evidence_package, {22: "amisulpride"})
        assert result["recall"] == 1.0
        assert result["points_covered"] == 2
        assert result["points_total"] == 2

    def test_compute_recall_partial(self, evidence_package):
        from psych_qa.evaluation.schemas import RequiredAnswerPoint
        points = [
            RequiredAnswerPoint(
                text="Mechanism involves D2 blockade",
                acceptable_evidence=[
                    AcceptableEvidence(source="stahl", category="mechanism", claim_hash="hash_b"),
                ],
            ),
            RequiredAnswerPoint(
                text="Something not in evidence",
                acceptable_evidence=[
                    AcceptableEvidence(source="stahl", category="mechanism", claim_hash="nonexistent"),
                ],
            ),
        ]
        result = compute_recall_at_k(points, evidence_package, {22: "amisulpride"})
        assert result["recall"] == 0.5
        assert result["points_covered"] == 1

    def test_compute_recall_empty_points(self, evidence_package):
        result = compute_recall_at_k([], evidence_package)
        assert result["recall"] == 1.0  # vacuously true
        assert result["points_total"] == 0


# ---------------------------------------------------------------------------
# Loader tests (using temp files)
# ---------------------------------------------------------------------------

class TestLoader:
    def test_load_empty_dataset(self, tmp_path):
        from psych_qa.evaluation import loader
        # Monkeypatch get_dataset_path
        original = loader.get_dataset_path
        loader.get_dataset_path = lambda name: tmp_path / f"{name}.jsonl"
        try:
            path = tmp_path / "test_empty.jsonl"
            path.write_text("")
            cases = loader.load_dataset("test_empty")
            assert cases == []
        finally:
            loader.get_dataset_path = original

    def test_load_golden_dataset(self, tmp_path):
        from psych_qa.evaluation import loader
        original = loader.get_dataset_path
        loader.get_dataset_path = lambda name: tmp_path / f"{name}.jsonl"
        try:
            path = tmp_path / "test_gold.jsonl"
            path.write_text(json.dumps({
                "question": "What is the mechanism of amisulpride?",
                "expected_status": "answered",
                "required_answer_points": [
                    {"text": "D2 blockade", "acceptable_evidence": [
                        {"source": "stahl", "category": "mechanism", "claim_hash": "abc"}
                    ]},
                ],
                "review_status": "needs_sasson_approval",
            }) + "\n")
            cases = loader.load_dataset("test_gold")
            assert len(cases) == 1
            assert isinstance(cases[0], GoldCase)
            assert cases[0].question == "What is the mechanism of amisulpride?"
            assert len(cases[0].required_answer_points) == 1
        finally:
            loader.get_dataset_path = original

    def test_filter_approved(self):
        from psych_qa.evaluation.loader import filter_approved, has_unapproved_cases
        cases = [
            GoldCase(question="q1", review_status=ReviewStatus.APPROVED),
            GoldCase(question="q2", review_status=ReviewStatus.NEEDS_SASSON_APPROVAL),
        ]
        approved = filter_approved(cases)
        assert len(approved) == 1
        assert approved[0].question == "q1"
        assert has_unapproved_cases(cases) is True

    def test_file_not_found(self):
        from psych_qa.evaluation import loader
        original = loader.get_dataset_path
        loader.get_dataset_path = lambda name: Path("/nonexistent") / f"{name}.jsonl"
        try:
            with pytest.raises(FileNotFoundError):
                loader.load_dataset("nonexistent")
        finally:
            loader.get_dataset_path = original
